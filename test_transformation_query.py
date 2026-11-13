#!/usr/bin/env python3
"""
Test transformation query against the devboard firmware.

Protocol (jetson_protocol.hpp):
  Send:    0x61 ('a' magic) + 0x74 ('t' query) + 1 byte frameDelay_ms
  Receive: EmbeddedTransformationMessage (73 bytes):
             magic (1B) + yaw (f32) + pitch (f32) + matrix[16] (f32x16)
"""

import argparse
import struct
import sys
import serial

JETSON_MESSAGE_MAGIC    = ord('a')
EMBEDDED_MESSAGE_MAGIC  = ord('b')
JETSON_TRANSFORMATION_QUERY = ord('t')

RESPONSE_SIZE = 1 + 4 + 4 + 16 * 4  # 73 bytes


def send_transformation_query(port: str, baud: int, frame_delay_ms: int, timeout: float):
    payload = bytes([JETSON_MESSAGE_MAGIC, JETSON_TRANSFORMATION_QUERY, frame_delay_ms])

    with serial.Serial(port, baud, timeout=timeout) as ser:
        ser.reset_input_buffer()
        ser.write(payload)
        print(f"Sent {len(payload)} bytes to {port}: {payload.hex(' ')}")

        response = ser.read(RESPONSE_SIZE)

    if len(response) < RESPONSE_SIZE:
        print(f"ERROR: expected {RESPONSE_SIZE} bytes, got {len(response)}")
        if response:
            print(f"  raw: {response.hex(' ')}")
        sys.exit(1)

    magic, yaw, pitch = struct.unpack_from('<Bff', response, 0)
    matrix = struct.unpack_from('<16f', response, 9)

    if magic != EMBEDDED_MESSAGE_MAGIC:
        print(f"WARNING: unexpected magic byte 0x{magic:02x} (expected 0x{EMBEDDED_MESSAGE_MAGIC:02x})")

    import math
    print(f"\nEmbeddedTransformationMessage ({RESPONSE_SIZE} bytes):")
    print(f"  magic : 0x{magic:02x} ({'OK' if magic == EMBEDDED_MESSAGE_MAGIC else 'BAD'})")
    print(f"  yaw   : {math.degrees(yaw):.3f} deg  ({yaw:.6f} rad)")
    print(f"  pitch : {math.degrees(pitch):.3f} deg  ({pitch:.6f} rad)")
    print(f"  4x4 camera->turret_field matrix (row-major):")
    for row in range(4):
        vals = matrix[row * 4:(row + 1) * 4]
        print(f"    [{', '.join(f'{v:8.4f}' for v in vals)}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Send transformation query to devboard')
    parser.add_argument('--port',    default='/dev/ttyTHS1', help='Serial port (default: /dev/ttyTHS1)')
    parser.add_argument('--baud',    default=460800, type=int, help='Baud rate (default: 460800)')
    parser.add_argument('--delay',   default=0, type=int, help='frameDelay_ms value (default: 0)')
    parser.add_argument('--timeout', default=2.0, type=float, help='Read timeout in seconds (default: 2.0)')
    args = parser.parse_args()

    send_transformation_query(args.port, args.baud, args.delay, args.timeout)
