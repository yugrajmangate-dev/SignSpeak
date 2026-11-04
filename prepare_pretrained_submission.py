"""Generate honest pretrained-demo submission documents from model metadata."""
import json
from PIL import Image, ImageDraw, ImageFont
from common import ROOT


def main() -> None:
    m = json.loads((ROOT / 'results/metrics.json').read_text())
    if m.get('data_source') != 'pretrained':
        raise SystemExit('Expected pretrained model metadata.')
    labels = ', '.join(m['labels'])
    metrics = '| Field | Status |\n| --- | --- |\n' + '\n'.join(
        f'| {name} | Not measured |' for name in ('Accuracy', 'Macro F1', 'Latency', 'Evaluation samples', 'Test session', 'Evaluation signers'))
    readme = f'''# SignSpeak — FT-04: Sign Language to Text/Speech Translator

Neural Storm 2026 · Future Tech. **Submission scope: a pretrained static gesture-to-text/speech demonstration, not full sign-language translation.**

## Team name and members

**YUGA** — Yugraj Mangate and Gayatri Karanjkar.

## Problem

FT-04 addresses communication barriers involving sign language. Complete translation needs language-specific vocabulary, motion, facial expression and grammar. We demonstrate only a small gesture-label-to-speech interface.

## Solution

The app uses the official pretrained MediaPipe Gesture Recognizer model, bundled in `models/gesture_recognizer.task`. No session recording or custom training is needed for this demo. We did not train the pretrained model or evaluate recognition accuracy on our own dataset.

Supported model labels: {labels}. These are gesture names, not alphabet predictions. Open_Palm is displayed as “Open Palm”, not translated to “HELLO”. ILoveYou is displayed as “I love you”.

## Pipeline

```mermaid
flowchart LR
    A[Camera image] --> B[MediaPipe hand landmarks]
    B --> C[Pretrained gesture classifier]
    C --> D[Gesture label]
    D --> E[Editable text]
    E --> F[Browser speech]
```

The Streamlit app processes individual snapshots. Desktop inference adds a majority vote and stability timer. Unrecognized or low-confidence poses produce no token.

## Tech stack

Python 3.11, MediaPipe Tasks, OpenCV, NumPy, Streamlit and browser Web Speech API. Desktop speech uses pyttsx3. The optional custom-training scripts use pandas, scikit-learn, joblib and matplotlib; those scripts are not required for the pretrained demo.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

For desktop webcam windows and speech:

```bash
python -m pip install -r requirements-local.txt
python -m pip install --force-reinstall --no-deps opencv-python==4.10.0.84
python infer.py
```

## Run locally

Take a camera photo while holding a supported pose. Read the predicted gesture and confidence, edit the sentence if desired, and click **Speak sentence**. Camera permission and browser speech support are required. Confidence is uncalibrated and is not accuracy.

For the local webcam script, ENTER speaks, C clears, Q quits. Remove your hand before repeating a token. Without a custom model.joblib, infer.py uses the bundled pretrained model automatically.

Optional custom-data workflow, separate from the submission demo:

```bash
python collect.py
python collect.py
python check_data.py
python train.py --signers 1
```

Do not claim this optional training workflow was completed unless you actually collect and evaluate data. Streamlit currently uses the pretrained bundle explicitly.

## Streamlit Community Cloud deployment

- Repository: yugrajmangate-dev/SignSpeak
- Branch: main
- Main file: app.py
- Advanced settings → Python version: **3.11**

The pinned MediaPipe cannot install on Python 3.14. Changing an existing deployment's Python version requires deleting that Cloud deployment and recreating it with Python 3.11; the GitHub repository remains intact. config.toml sets headless mode, not the Cloud Python interpreter. Cloud requirements exclude desktop audio and keyboard packages; packages.txt supplies Linux libraries required by MediaPipe's contrib OpenCV dependency. No API keys are needed.

## Screenshots

![Submission thumbnail](docs/thumbnail.png)

Real camera/demo screenshots remain to be captured. No confusion matrix is included because no project evaluation dataset exists.

## Metrics

{metrics}

`results/metrics.json` records data_source=pretrained, supported labels, the official model URL, file size and SHA-256. Evaluation fields are null. No held-out session or cross-signer evaluation has been performed. A model loading/sample inference check is not a recognition benchmark. We do not copy upstream benchmark scores into project results.

## Limitations and responsible use

Only supported static single-hand gestures are recognized. The upstream model card excludes full sign-language translation. The model does not recognize the requested alphabet vocabulary, motion, facial expression or grammar. No project-specific signer evaluation exists. It is not a replacement for human interpreters and should not be used for critical communication.

Snapshots are sent to the Streamlit server for in-memory processing and are not saved by the app. Optional collection saves landmarks and labels only, not images. Screenshots/video are separate user-created artifacts.

## AI tools used

AI tools used: Codex, Claude. They assisted implementation and planning. Pretrained weights are attributed to Google MediaPipe, not team YUGA.

## Model attribution

[Official model documentation](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer) · [Model card](https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf). See models/README.md and the included upstream license for source and provenance.
'''
    story = f'''## Inspiration

We are team YUGA—Yugraj Mangate and Gayatri Karanjkar. We wanted to explore a simple interface connecting visible hand gestures with text and speech, while being clear about the gap between gesture classification and full sign-language translation.

## What it does

Our Streamlit demo takes a camera snapshot, recognizes a supported static gesture, displays its label and confidence, and speaks editable text through the browser. Supported labels are {labels}. We do not translate an alphabet or complete signed sentences.

## How we built it

We integrated Google's official pretrained MediaPipe Gesture Recognizer with Python, OpenCV, NumPy, Streamlit and browser Web Speech API. The bundled model works without recording training sessions. Our optional desktop path uses a majority vote, stable-token text and pyttsx3 speech. Codex and Claude assisted implementation and planning. Google MediaPipe supplied the pretrained model; we do not claim authorship of its weights.

## Challenges we ran into

Our planned custom-data workflow required recordings that we did not complete. We switched to an attributed pretrained gesture model rather than claiming unmeasured results. Cloud logs also exposed a Python compatibility mismatch; deployment must use the pinned project's supported Python version.

## Accomplishments that we're proud of

We connected an existing gesture model to a compact text-and-speech interface and made the supported vocabulary and evaluation status explicit.

{metrics}

These fields are unmeasured in results/metrics.json. We have no project evaluation dataset, held-out test session or confusion matrix. Our sample inference checks do not establish accuracy or generalization.

## What we learned

We learned to separate model integration from training and validation. If we collect data later, we will split by session because random frame splits can leak nearly identical poses and inflate scores.

## Limitations and responsible use

We support a small set of static gestures, not full sign-language translation. The model omits language-specific motion and grammar. We performed no project-specific signer or cross-signer evaluation. The app does not store camera snapshots; optional collection stores landmarks and labels. It is not a replacement for human interpreters.

## What's next for SignSpeak

We would collect consented signer data, evaluate an untouched session, and investigate motion and language-specific vocabulary before making broader claims.
'''
    assert len(story.split()) < 600 and '[' not in story and ']' not in story
    docs = ROOT / 'docs'
    docs.mkdir(exist_ok=True)
    (ROOT / 'README.md').write_text(readme)
    (docs / 'devpost_story.md').write_text(story)
    (docs / 'DEMO_SCRIPT.md').write_text('''# SignSpeak screen-recording script

Target duration: 50 seconds. Timing is a filming plan, not a measured system result.

- 0–8 seconds: Show the app title. Say: “We are YUGA. SignSpeak is a pretrained static gesture-to-text and speech prototype.”
- 8–20 seconds: Hold Thumb_Up, take a photo, and show the actual displayed label and confidence. Say: “This uses Google's pretrained MediaPipe model. We did not train these weights.” Do not quote a confidence before observing it.
- 20–30 seconds: Try Open_Palm or Victory. Show whichever result actually appears. If recognition fails, show the failure rather than describing a success.
- 30–40 seconds: Show the editable text and click Speak sentence. Capture actual browser audio if available; otherwise state that audio was unavailable.
- 40–50 seconds: Open Model performance. Say: “Project accuracy and F1 have not been evaluated. This recognizes limited static gestures, not full sign language, and is not a replacement for interpreters.”

Use only the supported labels listed in results/metrics.json. Do not claim an accuracy, latency, dataset size, held-out session or cross-signer result. Remove notifications and personal information from the recording.
''')
    import matplotlib
    from pathlib import Path
    font_path = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
    font = lambda size: ImageFont.truetype(str(font_path), size)
    image = Image.new('RGB', (1500, 1000), '#0c1425')
    draw = ImageDraw.Draw(image)
    draw.text((85, 90), 'SignSpeak', font=font(115), fill='#66e1c4')
    draw.text((90, 245), 'Static gestures to text & speech', font=font(46), fill='white')
    draw.rounded_rectangle((85, 370, 1415, 720), radius=32, fill='#192943')
    draw.text((130, 420), 'Official pretrained MediaPipe model', font=font(43), fill='#66e1c4')
    draw.text((130, 510), 'Thumbs up  /  Open palm  /  Victory', font=font(41), fill='white')
    draw.text((130, 600), 'No custom training required', font=font(40), fill='#b7c8e3')
    draw.text((90, 800), 'Project accuracy & F1: not evaluated', font=font(40), fill='white')
    draw.text((90, 885), 'YUGA  |  Neural Storm 2026  |  FT-04 prototype', font=font(32), fill='#b7c8e3')
    image.save(docs / 'thumbnail.png', optimize=True)
    print('Generated pretrained README, story, demo script and thumbnail; no invented evaluation results.')


if __name__ == '__main__':
    main()
