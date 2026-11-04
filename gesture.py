"""Official MediaPipe pretrained gesture recognizer; no project training required."""
from pathlib import Path
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models/gesture_recognizer.task'
LABELS = ['Closed_Fist', 'Open_Palm', 'Pointing_Up', 'Thumb_Down', 'Thumb_Up', 'Victory', 'ILoveYou']


def create_recognizer() -> mp.tasks.vision.GestureRecognizer:
    options = mp.tasks.vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=1)
    return mp.tasks.vision.GestureRecognizer.create_from_options(options)


def predict(recognizer: mp.tasks.vision.GestureRecognizer, rgb: np.ndarray) -> tuple[str, float]:
    result = recognizer.recognize(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not result.gestures or not result.gestures[0]:
        return '', 0.0
    best = result.gestures[0][0]
    label = best.category_name or ''
    if label not in LABELS or best.score < .5:
        return '', float(best.score)
    return label, float(best.score)
