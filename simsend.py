import argparse
import json
import math
import random
import socket
import struct
import time

import cv2
import numpy as np

FRAME_W, FRAME_H = 960, 540
ICON_NAMES = ["sentry", "hero", "standard", "standard"]
PANEL_W, PANEL_H = 55, 25  # pixels (roughly proportional to real 12.5x5.5 cm)


class FakePanel:
    """A simulated armor panel that drifts smoothly across the frame."""

    def __init__(self):
        self.x = random.uniform(PANEL_W, FRAME_W - PANEL_W)
        self.y = random.uniform(PANEL_H, FRAME_H - PANEL_H)
        self.vx = random.uniform(-80, 80)
        self.vy = random.uniform(-60, 60)
        self.icon_id = random.randint(0, 3)
        self.depth_cm = random.uniform(100, 500)  # 1-5 meters in cm
        self.team_color = random.choice(["blue", "red"])

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce off edges
        if self.x < PANEL_W or self.x > FRAME_W - PANEL_W:
            self.vx *= -1
            self.x = max(PANEL_W, min(FRAME_W - PANEL_W, self.x))
        if self.y < PANEL_H or self.y > FRAME_H - PANEL_H:
            self.vy *= -1
            self.y = max(PANEL_H, min(FRAME_H - PANEL_H, self.y))

        # Slight random drift in velocity
        self.vx += random.uniform(-10, 10)
        self.vy += random.uniform(-10, 10)
        self.vx = max(-120, min(120, self.vx))
        self.vy = max(-100, min(100, self.vy))

        # Depth wanders slowly
        self.depth_cm += random.uniform(-5, 5)
        self.depth_cm = max(100, min(500, self.depth_cm))

    @property
    def bbx(self):
        x1 = int(self.x - PANEL_W / 2)
        y1 = int(self.y - PANEL_H / 2)
        return [x1, y1, PANEL_W, PANEL_H]

    @property
    def tvec(self):
        # Fake tvec in cm: X offset from center, Y is up/down, Z is depth
        x_cm = (self.x - FRAME_W / 2) * self.depth_cm / 500
        y_cm = (self.y - FRAME_H / 2) * self.depth_cm / 500
        return [x_cm, y_cm, self.depth_cm]

    def to_dict(self):
        tv = self.tvec
        return {
            "icon": ICON_NAMES[self.icon_id],
            "icon_id": self.icon_id,
            "bbx": self.bbx,
            "tvec": tv,
            "rvec": [random.uniform(-0.3, 0.3) for _ in range(3)],
            "team": self.team_color,
        }

    def mcu_coords(self):
        """Coordinate transform matching communicate.py lines 87-91."""
        tv = self.tvec
        return {
            "X": tv[0] / 100.0,
            "Y": tv[2] / 100.0,
            "Z": -tv[1] / 100.0,
        }


def draw_panel(frame, panel, is_target):
    bbx = panel.bbx
    x1, y1, w, h = bbx
    x2, y2 = x1 + w, y1 + h

    # Panel fill color
    color = (255, 100, 50) if panel.team_color == "blue" else (50, 50, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

    # Border
    border_color = (0, 255, 0) if is_target else (200, 200, 200)
    thickness = 3 if is_target else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness)

    # Icon label
    label = ICON_NAMES[panel.icon_id]
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if is_target:
        cv2.putText(frame, "TARGET", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Fake robot simulator - streams synthetic detections")
    parser.add_argument("--port", type=int, default=5555, help="Port to listen on (default: 5555)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--panels", type=int, default=3, help="Number of fake panels (default: 3)")
    args = parser.parse_args()

    # Create fake panels
    panels = [FakePanel() for _ in range(args.panels)]

    # TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"Fake robot listening on 0.0.0.0:{args.port}")

    frame_interval = 1.0 / args.fps

    try:
        while True:
            print("Waiting for viewer connection...")
            conn, addr = server.accept()
            print(f"Viewer connected from {addr}")

            last_time = time.time()
            try:
                while True:
                    now = time.time()
                    dt = now - last_time
                    last_time = now

                    # Update panel positions
                    for p in panels:
                        p.update(dt)

                    # Select target (closest panel)
                    target_idx = min(range(len(panels)), key=lambda i: panels[i].depth_cm)

                    # Render frame
                    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
                    frame[:] = (30, 30, 30)  # dark gray background

                    # Draw crosshair at center
                    cx, cy = FRAME_W // 2, FRAME_H // 2
                    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (60, 60, 60), 1)
                    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (60, 60, 60), 1)

                    for i, p in enumerate(panels):
                        draw_panel(frame, p, i == target_idx)

                    # Overlay target XYZ
                    target = panels[target_idx]
                    mcu = target.mcu_coords()
                    xyz_text = f"X:{mcu['X']:.3f} Y:{mcu['Y']:.3f} Z:{mcu['Z']:.3f} m"
                    cv2.putText(frame, xyz_text, (10, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                    # Encode frame
                    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    jpeg_bytes = jpeg.tobytes()

                    # Build metadata
                    metadata = {
                        "timestamp": now,
                        "panels": [p.to_dict() for p in panels],
                        "target_idx": target_idx,
                        "mcu": mcu,
                    }
                    meta_bytes = json.dumps(metadata).encode()

                    # Send: [total_len][jpeg_len][jpeg][meta]
                    jpeg_len = len(jpeg_bytes)
                    total_len = 4 + jpeg_len + len(meta_bytes)
                    header = struct.pack(">II", total_len, jpeg_len)
                    conn.sendall(header + jpeg_bytes + meta_bytes)

                    # Throttle to target FPS
                    elapsed = time.time() - now
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except (BrokenPipeError, ConnectionResetError, OSError):
                print("Viewer disconnected")
                conn.close()
    except KeyboardInterrupt:
        print("\nShutting down")
    finally:
        server.close()


if __name__ == "__main__":
    main()
