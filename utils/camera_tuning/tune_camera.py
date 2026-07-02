#!/usr/bin/env python3
"""GUI to live-tune V4L2 camera controls (brightness, contrast, exposure, ...).

Standalone utility -- no AutoX imports. It discovers whatever controls the
camera exposes via ``v4l2-ctl --list-ctrls-menus``, builds an input box (with its
valid ``[min, max]`` range shown) / checkbox / dropdown for each, applies changes
live with ``v4l2-ctl --set-ctrl``, and shows a live preview so you can see the
effect. Type a value and press Enter (or click away) to apply it. This is the same
control mechanism the
camera driver uses for manual exposure (``src/drivers/video_stream.py``), so the
values you find here are exactly what the robot will apply.

V4L2 (USB UVC) cameras only -- it shells out to ``v4l2-ctl``.

Usage:
    uv run python utils/camera_tuning/tune_camera.py
    uv run python utils/camera_tuning/tune_camera.py --device /dev/video0 \
        --width 1280 --height 720 --fps 90
    uv run python utils/camera_tuning/tune_camera.py --no-preview

Buttons:
    Reset defaults  -- set every (active) control back to its V4L2 default.
    Export settings -- write a v4l2-ctl shell command + an info.yaml snippet to
                       utils/camera_tuning/camera_settings.txt (also printed on quit).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Dict, List, Optional

EXPORT_PATH = Path(__file__).resolve().parent / "camera_settings.txt"

# A control line, e.g.:
#   brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0 flags=...
_CTRL_RE = re.compile(
    r"^\s*(?P<name>\w+)\s+0x[0-9a-fA-F]+\s+\((?P<type>\w+)\)\s*:\s*(?P<rest>.*)$"
)
# A menu item line under a (menu)/(intmenu) control, e.g. "    1: Manual Mode".
_MENU_RE = re.compile(r"^\s+(?P<idx>-?\d+):\s*(?P<label>.*)$")


class Control:
    """One V4L2 control parsed from ``v4l2-ctl --list-ctrls-menus``."""

    def __init__(self, name: str, ctype: str, fields: Dict[str, str]):
        self.name = name
        self.type = ctype  # int, bool, menu, intmenu, int64, button, string
        self.min = _to_int(fields.get("min"))
        self.max = _to_int(fields.get("max"))
        self.step = _to_int(fields.get("step")) or 1
        self.default = _to_int(fields.get("default"))
        self.value = _to_int(fields.get("value"))
        self.flags = fields.get("flags", "")
        self.menu: Dict[int, str] = {}  # idx -> label, for menu/intmenu

    @property
    def inactive(self) -> bool:
        """True if the control is currently read-only/inactive.

        e.g. exposure_time_absolute is inactive while auto_exposure != Manual.
        """
        return "inactive" in self.flags or "read-only" in self.flags

    @property
    def tunable(self) -> bool:
        """True if this control type gets a widget (int/bool/menu/intmenu)."""
        return self.type in ("int", "bool", "menu", "intmenu")


def _to_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def list_devices() -> List[str]:
    """Return /dev/video* paths that report controls (best-effort)."""
    return sorted(str(p) for p in Path("/dev").glob("video*"))


def query_controls(device: str) -> List[Control]:
    """Parse ``v4l2-ctl --list-ctrls-menus`` into Control objects."""
    out = subprocess.run(
        ["v4l2-ctl", "-d", device, "--list-ctrls-menus"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"cannot open {device}")

    controls: List[Control] = []
    current: Optional[Control] = None
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        m = _CTRL_RE.match(line)
        if m:
            rest = m.group("rest")
            fields = dict(re.findall(r"(\w+)=([-\w]+)", rest))
            current = Control(m.group("name"), m.group("type"), fields)
            controls.append(current)
            continue
        mm = _MENU_RE.match(line)
        if mm and current is not None:
            current.menu[int(mm.group("idx"))] = mm.group("label").strip()
    return controls


def set_control(device: str, name: str, value: int) -> Optional[str]:
    """Apply one control; return an error string on failure, else None."""
    out = subprocess.run(
        ["v4l2-ctl", "-d", device, "--set-ctrl", f"{name}={value}"],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return out.stderr.strip() or "set failed"
    return None


class PreviewWorker:
    """Background camera reader (OpenCV) holding only the newest BGR frame."""

    def __init__(self, device: str, width: int, height: int, fps: int):
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self._latest = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self) -> None:
        """Spawn the background capture thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import cv2

        idx = int(re.sub(r"\D", "", self.device) or 0)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            self.error = f"OpenCV could not open {self.device}"
            return
        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = frame
        cap.release()

    def latest(self):
        """Return the most recently decoded BGR frame, or None."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        """Signal the capture thread to stop and join it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class TunerApp:
    """Tkinter app: a control panel of V4L2 sliders/menus beside a live preview."""

    def __init__(self, device: str, width: int, height: int, fps: int, preview: bool):
        self.device = device
        self.controls = query_controls(device)
        self.root = tk.Tk()
        self.root.title(f"Camera tuner -- {device}")
        self.widgets: Dict[str, Dict] = {}  # name -> {var, widget, control}

        self._build_ui()

        self.preview: Optional[PreviewWorker] = None
        if preview:
            self.preview = PreviewWorker(device, width, height, fps)
            self.preview.start()
            self.root.after(50, self._update_preview)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        # Left: scrollable controls panel.
        left = ttk.Frame(outer)
        left.grid(row=0, column=0, sticky="ns")
        canvas = tk.Canvas(left, width=470, highlightthickness=0)
        scroll = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self._ctrl_frame = ttk.Frame(canvas)
        self._ctrl_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self._ctrl_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"),
        )
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        tunable = [c for c in self.controls if c.tunable]
        if not tunable:
            ttk.Label(self._ctrl_frame, text="No tunable V4L2 controls found.").pack(
                anchor="w"
            )
        for ctrl in tunable:
            self._add_control_row(ctrl)

        # Right: preview.
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)
        self._preview_label = ttk.Label(right, text="(preview loading...)")
        self._preview_label.pack(fill="both", expand=True)

        # Bottom: buttons + status.
        bar = ttk.Frame(outer, padding=(0, 8, 0, 0))
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(bar, text="Reset defaults", command=self._reset_defaults).pack(
            side="left"
        )
        ttk.Button(bar, text="Refresh", command=self._refresh).pack(side="left", padx=4)
        ttk.Button(bar, text="Export settings", command=self._export).pack(side="left")
        self._status = ttk.Label(bar, text="")
        self._status.pack(side="right")

    def _add_control_row(self, ctrl: Control) -> None:
        row = ttk.Frame(self._ctrl_frame, padding=(0, 3))
        row.pack(fill="x", anchor="w")
        ttk.Label(row, text=ctrl.name, width=26, anchor="w").pack(side="left")

        if ctrl.type == "bool":
            var = tk.IntVar(value=ctrl.value or 0)
            w = ttk.Checkbutton(
                row, variable=var, command=lambda c=ctrl: self._on_change(c)
            )
            w.pack(side="left")
        elif ctrl.type in ("menu", "intmenu"):
            var = tk.StringVar()
            labels = {f"{i}: {lab}": i for i, lab in ctrl.menu.items()}
            cur_label = next(
                (k for k, i in labels.items() if i == ctrl.value),
                next(iter(labels), ""),
            )
            var.set(cur_label)
            w = ttk.Combobox(
                row, textvariable=var, values=list(labels), state="readonly", width=22
            )
            w.bind("<<ComboboxSelected>>", lambda e, c=ctrl: self._on_change(c))
            w.pack(side="left")
            self.widgets[ctrl.name] = {
                "var": var,
                "widget": w,
                "control": ctrl,
                "labels": labels,
            }
            self._set_state(ctrl)
            return
        else:  # int / int64 -- a typed input box plus a range hint
            var = tk.IntVar(value=ctrl.value if ctrl.value is not None else 0)
            entry = ttk.Entry(row, width=8, justify="right")
            entry.insert(0, str(var.get()))
            entry.pack(side="left")
            # Commit a typed value on Enter or when focus leaves the box.
            entry.bind("<Return>", lambda e, c=ctrl: self._on_entry(c))
            entry.bind("<FocusOut>", lambda e, c=ctrl: self._on_entry(c))
            ttk.Label(row, text=self._range_hint(ctrl), foreground="#666").pack(
                side="left", padx=(6, 0)
            )
            self.widgets[ctrl.name] = {
                "var": var,
                "widget": entry,  # the input box is the control's main widget
                "control": ctrl,
                "display": entry,
            }
            self._set_state(ctrl)
            return

        self.widgets[ctrl.name] = {"var": var, "widget": w, "control": ctrl}
        self._set_state(ctrl)

    @staticmethod
    def _range_hint(ctrl: Control) -> str:
        """A '[min, max] (default N)' hint shown next to an int input box."""
        lo = ctrl.min if ctrl.min is not None else "?"
        hi = ctrl.max if ctrl.max is not None else "?"
        text = f"[{lo}, {hi}]"
        if ctrl.default is not None:
            text += f"  default {ctrl.default}"
        if ctrl.step and ctrl.step != 1:
            text += f"  step {ctrl.step}"
        return text

    def _set_entry(self, ctrl: Control, value: int) -> None:
        """Replace the text shown in an int control's entry box."""
        entry = self.widgets[ctrl.name].get("display")
        if entry is None:
            return
        entry.delete(0, tk.END)
        entry.insert(0, str(value))

    def _set_state(self, ctrl: Control) -> None:
        entry = self.widgets.get(ctrl.name)
        if not entry:
            return
        state = "disabled" if ctrl.inactive else "normal"
        try:
            if isinstance(entry["widget"], ttk.Combobox):
                entry["widget"].configure(
                    state="disabled" if ctrl.inactive else "readonly"
                )
            else:
                entry["widget"].configure(state=state)
            if entry.get("display") is not None:
                entry["display"].configure(state=state)
        except tk.TclError:
            pass

    # -------------------------------------------------------------- events --
    def _on_entry(self, ctrl: Control) -> None:
        """Commit a value typed into an int control's input box."""
        entry = self.widgets[ctrl.name]
        raw = entry["display"].get().strip()
        try:
            value = int(round(float(raw)))
        except ValueError:
            # Not a number -- restore the last good value and bail.
            self._set_entry(ctrl, entry["var"].get())
            self._status.configure(
                text=f"{ctrl.name}: invalid number", foreground="red"
            )
            return
        if ctrl.min is not None:
            value = max(ctrl.min, value)
        if ctrl.max is not None:
            value = min(ctrl.max, value)
        entry["var"].set(value)
        self._set_entry(ctrl, value)  # reflect any clamping
        self._on_change(ctrl)

    def _on_change(self, ctrl: Control) -> None:
        entry = self.widgets[ctrl.name]
        if "labels" in entry:  # menu
            value = entry["labels"].get(entry["var"].get())
        else:
            value = int(entry["var"].get())
        if value is None:
            return
        err = set_control(self.device, ctrl.name, value)
        if err:
            self._status.configure(text=f"{ctrl.name}: {err}", foreground="red")
            return
        ctrl.value = value
        self._status.configure(text=f"set {ctrl.name}={value}", foreground="green")
        # Changing one control (e.g. auto_exposure) can (de)activate others.
        self._refresh(rebuild=False)

    def _reset_defaults(self) -> None:
        for ctrl in self.controls:
            if ctrl.tunable and not ctrl.inactive and ctrl.default is not None:
                set_control(self.device, ctrl.name, ctrl.default)
        self._refresh(rebuild=False)
        self._status.configure(text="reset to defaults", foreground="green")

    def _refresh(self, rebuild: bool = True) -> None:
        """Re-read control values/flags from the device and sync widgets."""
        try:
            fresh = {c.name: c for c in query_controls(self.device)}
        except RuntimeError as e:
            self._status.configure(text=str(e), foreground="red")
            return
        for name, entry in self.widgets.items():
            new = fresh.get(name)
            if new is None:
                continue
            entry["control"].value = new.value
            entry["control"].flags = new.flags
            if "labels" in entry:
                for k, i in entry["labels"].items():
                    if i == new.value:
                        entry["var"].set(k)
                        break
            elif new.value is not None:
                entry["var"].set(new.value)
                if entry.get("display") is not None:
                    self._set_entry(entry["control"], new.value)
            self._set_state(entry["control"])

    # ------------------------------------------------------------- preview --
    def _update_preview(self) -> None:
        if self.preview is None:
            return
        if self.preview.error:
            self._preview_label.configure(text=self.preview.error, image="")
            return
        frame = self.preview.latest()
        if frame is not None:
            import cv2
            from PIL import Image, ImageTk

            h, w = frame.shape[:2]
            scale = min(720 / w, 540 / h, 1.0)
            disp = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._preview_label.configure(image=img, text="")
            self._preview_label.image = img  # keep a reference
        self.root.after(33, self._update_preview)

    # -------------------------------------------------------------- export --
    def _export(self) -> None:
        text = self._settings_text()
        EXPORT_PATH.write_text(text)
        self._status.configure(text=f"wrote {EXPORT_PATH.name}", foreground="green")
        print(text)

    def _settings_text(self) -> str:
        ctrls = sorted(self.widgets.values(), key=lambda e: e["control"].name)
        set_args = " ".join(
            f"--set-ctrl {e['control'].name}={e['control'].value}"
            for e in ctrls
            if e["control"].value is not None
        )
        lines = [
            f"# Camera settings tuned on {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# device: {self.device}",
            "",
            "# Apply all controls at once:",
            f"v4l2-ctl -d {self.device} {set_args}",
            "",
            "# Per-control values:",
        ]
        for e in ctrls:
            c = e["control"]
            lines.append(f"#   {c.name} = {c.value}")
        # info.yaml only models exposure today; surface it explicitly.
        exp = next(
            (
                e["control"].value
                for e in ctrls
                if e["control"].name == "exposure_time_absolute"
            ),
            None,
        )
        if exp is not None:
            lines += [
                "",
                "# info.yaml (src/info.yaml hardware block) exposure field:",
                f"#   camera_exposure: {exp}",
            ]
        return "\n".join(lines) + "\n"

    # --------------------------------------------------------------- close --
    def _on_close(self) -> None:
        if self.preview is not None:
            self.preview.stop()
        print(self._settings_text())
        self.root.destroy()

    def run(self) -> None:
        """Enter the Tk main loop (blocks until the window closes)."""
        self.root.mainloop()


def main() -> int:
    """Parse args, open the device, and run the tuner GUI."""
    ap = argparse.ArgumentParser(description=__doc__)
    devs = list_devices()
    ap.add_argument(
        "--device",
        default=devs[0] if devs else "/dev/video0",
        help="V4L2 device (default: first /dev/video*)",
    )
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=90)
    ap.add_argument("--no-preview", action="store_true", help="disable live preview")
    args = ap.parse_args()

    if not Path(args.device).exists():
        print(f"error: {args.device} does not exist.", file=sys.stderr)
        if devs:
            print(f"available: {', '.join(devs)}", file=sys.stderr)
        else:
            print(
                "no /dev/video* devices found -- is a camera plugged in?",
                file=sys.stderr,
            )
        return 1

    try:
        app = TunerApp(
            args.device, args.width, args.height, args.fps, not args.no_preview
        )
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
