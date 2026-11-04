"""Webcam: 10-frame vote, one-second stable tokens, ENTER speaks, Q quits."""
import time
from collections import Counter, deque
import cv2
import joblib
import mediapipe as mp
from common import ROOT, normalize
from gesture import create_recognizer, predict


def main() -> None:
    custom = (ROOT / 'model.joblib').exists()
    bundle = joblib.load(ROOT / 'model.joblib') if custom else {}
    if bundle.get('synthetic'):
        print('SYNTHETIC model: predictions are not real recognition. Collect sessions and retrain.')
    model = bundle.get('model')
    if not custom:
        print('Official pretrained gesture model; no project accuracy evaluation.')
    try:
        import pyttsx3
        voice = pyttsx3.init()
    except Exception as error:
        voice = None
        print(f'Speech unavailable: {error}')
    votes: deque[str] = deque(maxlen=10)
    sentence, stable, appended, since = '', '', '', 0.0
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam 0; check permissions.')
        with (mp.solutions.hands.Hands(max_num_hands=1) if custom else create_recognizer()) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if custom:
                    result = hands.process(rgb)
                    raw = ''
                    if result.multi_hand_landmarks:
                        features = normalize(result.multi_hand_landmarks[0].landmark, result.multi_handedness[0].classification[0].label)
                        if features is not None:
                            raw = str(model.predict(features.reshape(1, -1))[0])
                else:
                    raw, _ = predict(hands, rgb)
                label = ''
                if raw:
                    votes.append(raw)
                    if len(votes) == 10:
                        candidate, count = Counter(votes).most_common(1)[0]
                        label = candidate if count > 5 else ''
                else:
                    votes.clear()
                    appended = ''
                now = time.monotonic()
                if label != stable or not label:
                    stable, since = label, now
                elif now - since >= 1 and label != appended:
                    sentence += ('' if len(label) == 1 else ' ') + ('I love you' if label == 'ILoveYou' else label.replace('_', ' '))
                    if len(label) > 1:
                        sentence += ' '
                    appended = label
                cv2.putText(frame, f'Sign: {label or "..."}', (10, 35), 0, .8, (0, 255, 0), 2)
                cv2.putText(frame, sentence[-65:], (10, frame.shape[0] - 45), 0, .6, (255, 255, 255), 2)
                cv2.putText(frame, 'ENTER speak | C clear | Q quit', (10, frame.shape[0] - 15), 0, .5, (255, 255, 255), 1)
                cv2.imshow('SignSpeak - SYNTHETIC' if bundle.get('synthetic') else 'SignSpeak', frame)
                key = cv2.waitKey(1) & 255
                if key == ord('q'):
                    break
                if key == ord('c'):
                    sentence, stable, appended, since = '', '', '', now
                    votes.clear()
                if key in (10, 13) and voice and sentence.strip():
                    voice.say(sentence.strip())
                    voice.runAndWait()
                    votes.clear()
                    stable, since = '', time.monotonic()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
