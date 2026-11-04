"""Number keys choose a label; hold SPACE to record; Q quits."""
import argparse
import csv
from threading import Event
import cv2
import mediapipe as mp
from pynput import keyboard
from common import ROOT, FEATURES, LABELS, normalize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", default=LABELS)
    labels = parser.parse_args().labels
    if not 1 <= len(labels) <= 10:
        parser.error("Supply 1–10 labels for keys 0–9.")
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    n = max((int(p.stem.split('_')[-1]) for p in folder.glob('session_*.csv')), default=-1) + 1
    path = folder / f"session_{n}.csv"
    held = Event()
    # Key-up events are needed: OpenCV alone only provides key presses/repeats.
    def press(key: keyboard.Key) -> None:
        if key == keyboard.Key.space:
            held.set()
    def release(key: keyboard.Key) -> None:
        if key == keyboard.Key.space:
            held.clear()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open webcam 0; check camera permissions.")
    selected, count = 0, 0
    print(f"Saving {path}; " + " | ".join(f"{i}:{v}" for i, v in enumerate(labels)))
    try:
        with keyboard.Listener(on_press=press, on_release=release), path.open('x', newline='') as file, mp.solutions.hands.Hands(max_num_hands=1) as hands:
            writer = csv.writer(file)
            writer.writerow(FEATURES + ['label'])
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if held.is_set() and result.multi_hand_landmarks:
                    features = normalize(result.multi_hand_landmarks[0].landmark, result.multi_handedness[0].classification[0].label)
                    if features is not None:
                        writer.writerow([*features, labels[selected]])
                        file.flush()
                        count += 1
                cv2.putText(frame, f"{labels[selected]} | SPACE={held.is_set()} | saved={count}", (10, 30), 0, .7, (0, 255, 0), 2)
                cv2.imshow('SignSpeak collection | 0-9 select | Q quit', frame)
                key = cv2.waitKey(1) & 255
                if key == ord('q'):
                    break
                if ord('0') <= key < ord('0') + len(labels):
                    selected = key - ord('0')
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
