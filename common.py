"""Shared feature order: x0,y0,...,x20,y20; inputs must be selfie-mirrored."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
FEATURES = [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]
LABELS = ["A", "B", "C", "L", "V", "W", "Y", "OK", "THUMBS_UP", "HELLO"]


def normalize(landmarks: list, handedness: str) -> np.ndarray | None:
    points = np.array([[p.x, p.y] for p in landmarks], dtype=np.float32)
    points -= points[0]
    scale = np.linalg.norm(points, axis=1).max()
    if scale < 1e-6:
        return None
    points /= scale
    if handedness == "Left":
        points[:, 0] *= -1
    return points.flatten()
