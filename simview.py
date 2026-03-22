import argparse
import json
import socket
import struct
import time

import cv2
import numpy as np


def recv_exact(sock, n):
    """Receive exactly n bytes from socket."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data.extend(chunk)
    return bytes(data)


def draw_overlays(frame, metadata):
    """Draw detection overlays from metadata onto the frame."""
    panels = metadata.get("panels", [])
    target_idx = metadata.get("target_idx", -1)
    mcu = metadata.get("mcu", {})

    for i, panel in enumerate(panels):
        bbx = panel["bbx"]
        x1, y1, w, h = bbx
        x2, y2 = x1 + w, y1 + h
        is_target = (i == target_idx)

        # Bounding box
        color = (0, 255, 0) if is_target else (0, 200, 255)
        thickness = 2 if is_target else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label
        icon = panel.get("icon", "?")
        team = panel.get("team", "?")
        label = f"{icon} ({team})"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if is_target:
            cv2.putText(frame, "TARGET", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # MCU coordinates
    if mcu:
        xyz = f"MCU -> X:{mcu.get('X', 0):.3f} Y:{mcu.get('Y', 0):.3f} Z:{mcu.get('Z', 0):.3f} m"
        cv2.putText(frame, xyz, (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="View live stream from fake robot simulator")
    parser.add_argument("host", help="Robot IP address (e.g. 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5555, help="Port to connect to (default: 5555)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to {args.host}:{args.port}...")
    sock.connect((args.host, args.port))
    print("Connected")

    fps_time = time.time()
    frame_count = 0
    fps = 0.0

    try:
        while True:
            # Receive header: [total_len (4B)][jpeg_len (4B)]
            header = recv_exact(sock, 8)
            total_len, jpeg_len = struct.unpack(">II", header)

            # total_len includes the 4-byte jpeg_len field, so payload is total_len - 4
            payload = recv_exact(sock, total_len - 4)
            jpeg_bytes = payload[:jpeg_len]
            meta_bytes = payload[jpeg_len:]

            # Decode frame
            frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Parse metadata
            metadata = json.loads(meta_bytes)

            # Draw detection overlays
            draw_overlays(frame, metadata)

            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("AutoX Sim View", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except (ConnectionError, struct.error):
        print("Connection lost")
    except KeyboardInterrupt:
        print("\nClosing")
    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
