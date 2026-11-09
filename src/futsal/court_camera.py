"""Pan/tilt/zoom camera model for fixed-position broadcast cameras viewing the 40x20 m court plane.

Court frame: x along the length, y toward the near touchline, z DOWN (right-handed), camera above the
court so its z coordinate is negative.  Image: x right, y down, principal point at the image centre.
"""
import cv2
import numpy as np


def look_at(C, target, roll=0.0):
    C = np.asarray(C, float)
    fwd = np.asarray(target, float) - C
    fwd /= np.linalg.norm(fwd)
    down_hint = np.array([0.0, 0.0, 1.0])
    right = np.cross(down_hint, fwd)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        right, down = c * right + s * down, -s * right + c * down
    return np.vstack([right, down, fwd])


def homography_from_camera(C, target, f, size, roll=0.0):
    """court(x,y) -> image pixel homography for a camera at C aimed at the court point `target`."""
    w, h = size
    R = look_at(C, np.array([target[0], target[1], 0.0]), roll)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    C = np.asarray(C, float)
    M = np.column_stack([R[:, 0], R[:, 1], -R @ C])
    return K @ M


def decompose_homography(H, size):
    """Return (f, R, C) from a court->image homography assuming square pixels, centred principal point."""
    w, h = size
    cx, cy = w / 2.0, h / 2.0
    A = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]]) @ np.asarray(H, float)  # = diag(f,f,1) [r1 r2 t] * s
    h1, h2, h3 = A[:, 0], A[:, 1], A[:, 2]
    # r1.r2 = 0 -> h1x h2x/f^2 + h1y h2y/f^2 + h1z h2z = 0
    den = h1[2] * h2[2]
    if abs(den) < 1e-12:
        return None
    f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den
    if f2 <= 0:
        return None
    f = float(np.sqrt(f2))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    r1, r2, t = Kinv @ h1, Kinv @ h2, Kinv @ h3
    s = 1.0 / np.linalg.norm(r1)
    r1, r2, t = r1 * s, r2 * s, t * s
    if t[2] < 0:  # points must lie in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    C = -R.T @ t
    return f, R, C
