"""10-state full-robot EKF, ported from TAMU-Robomasters/Full-Robot-EKF @5672b45.

State lives in the turret frame (x right, y forward, z up, metres); the camera
extrinsic enters only through the measurement model ``h(x)`` (see ``model.py``).
"""
