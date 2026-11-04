"""Generate submission docs and thumbnail only from recorded evaluation artifacts."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cross-signer', action='store_true', help='Use only for a confirmed unseen-signer test session')
    args = parser.parse_args()
    path = ROOT / 'results/metrics.json'
    if not path.exists():
        raise SystemExit('STOP: recorded results/metrics.json is missing. Collect data and train first; no docs or thumbnail generated.')
    m = json.loads(path.read_text())
    required = ('data_source', 'accuracy', 'macro_f1', 'latency_ms', 'n_samples', 'test_session', 'signers', 'n_sessions', 'train_size', 'test_size', 'labels', 'model_chosen')
    missing = [key for key in required if key not in m]
    if missing or m.get('data_source') != 'recorded':
        raise SystemExit(f'STOP: recorded metrics required; missing fields: {missing}')
    matrix = ROOT / 'results/confusion_matrix.png'
    if not matrix.exists():
        raise SystemExit('STOP: results/confusion_matrix.png is missing.')
    if args.cross_signer and m['signers'] < 2:
        raise SystemExit('STOP: cross-signer evaluation requires multiple recorded signers.')
    table = '\n'.join(['| Metric | Recorded result |', '| --- | --- |',
        f"| Accuracy | {m['accuracy']:.2%} |", f"| Macro F1 | {m['macro_f1']:.4f} |",
        f"| Model-only latency | {m['latency_ms']:.3f} ms/sample |",
        f"| Samples | {m['n_samples']} |", f"| Test session | {m['test_session']} |",
        f"| Signers | {m['signers']} |"])
    split = ('The last session is a confirmed cross-signer test: its signer is absent from training. '
             if args.cross_signer else 'This is a session-held-out evaluation; cross-signer generalization is not claimed. ')
    split += 'The last session also selects the winning model, so this is not an independent final test. Random frame splits can leak near-identical poses between training and evaluation.'
    readme = f'''# SignSpeak — FT-04: Sign Language to Text/Speech Translator

Neural Storm 2026 · Track: Future Tech. A limited-vocabulary, static hand-pose prototype.

## Team name and members

**YUGA** — Yugraj Mangate and Gayatri Karanjkar.

## Problem

FT-04 explores communication barriers between signers and people unfamiliar with signing. Full sign-language translation includes motion, expression and grammar; this project addresses only isolated hand poses.

## Solution

A local webcam collects landmarks and labels. A classifier predicts a supported pose, a majority vote smooths predictions, and stable tokens build text that can be spoken. Streamlit provides snapshot prediction, an uncalibrated confidence bar, performance results and browser speech. Supported labels: {', '.join(m['labels'])}.

## Pipeline

```mermaid
flowchart LR
    A[Webcam] --> B[MediaPipe 21 landmarks]
    B --> C[Normalization]
    C --> D[Classifier]
    D --> E[Majority vote]
    E --> F[Text/speech]
```

Normalization subtracts the wrist, scales by hand size, mirrors left hands and flattens x/y coordinates. Local inference uses the configured vote window and stability timer; the Streamlit snapshot path predicts each photo directly.

## Tech stack

Python 3.11, MediaPipe, OpenCV, NumPy, pandas, scikit-learn, joblib, matplotlib, pyttsx3, pynput, Streamlit and browser Web Speech API. Direct dependencies are pinned.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
python -m pip install --force-reinstall --no-deps opencv-python==4.10.0.84
```

Grant camera access and macOS Input Monitoring/Accessibility access for SPACE tracking. Local Linux speech may require espeak. Cloud dependencies exclude pyttsx3 and include headless OpenCV; legacy MediaPipe also pulls in contrib OpenCV, so packages.txt supplies Linux runtime libraries. On Community Cloud, select app.py and Python 3.11, and include the trusted model and results in the repository.

## Run locally

```bash
python collect.py
python collect.py
python check_data.py
python train.py --signers {m['signers']}
python infer.py
streamlit run app.py
```

Use number keys to select labels and hold SPACE to record. Use the same label set in every session; vary distance, angle or lighting for the final session. In inference, ENTER speaks, C clears and Q quits. Remove your hand before repeating an identical token. The Streamlit sentence field is editable; its Speak sentence button uses browser audio.

## Screenshots

![Local webcam demo](docs/screenshots/live_demo.png)
![Streamlit demo](docs/screenshots/streamlit_app.png)
![Held-out confusion matrix](results/confusion_matrix.png)

## Metrics

{table}

Selected model: {m['model_chosen']}. Training samples: {m['train_size']}; held-out samples: {m['test_size']}; recorded sessions: {m['n_sessions']}. Results and per-class reports come from results/metrics.json. Latency measures warmed-up single-sample model prediction, excluding capture, landmark extraction, UI and speech.

{split}

## Limitations and responsible use

Static signs only, a small vocabulary and {m['signers']} recorded signer(s) limit generalization. Single-hand x/y features omit motion, depth, facial expression and grammar; HELLO is a held-pose shortcut. Unknown poses still get known labels. Confidence is uncalibrated. This is not a replacement for human interpreters or a tool for critical communication.

Collection stores landmarks and labels only; no images are stored. Streamlit snapshots are sent to the server for in-memory processing and are not saved by the app. Demo screenshots/video are separate, intentionally created artifacts.

## AI tools used

AI tools used: Codex, Claude. These tools assisted development and planning; measured performance is derived from recorded session files.
'''
    story = f'''## Inspiration

We built SignSpeak as team YUGA—Yugraj Mangate and Gayatri Karanjkar—to explore communication through a small set of isolated hand poses. We kept the scope narrow enough to inspect honestly.

## What it does

We map supported static hand poses to text. Desktop inference smooths predictions, builds a sentence and speaks on request. Our Streamlit app predicts camera snapshots, displays confidence and speaks editable text using browser audio.

## How we built it

We used Python, MediaPipe landmarks, shared wrist/scale/handedness normalization, OpenCV, scikit-learn, Streamlit and local or browser speech. We trained RandomForest and a small MLP and selected {m['model_chosen']} by held-out macro F1, breaking ties by accuracy. We used Codex and Claude for development assistance.

## Challenges we ran into

We separated recording sessions because random frame splits can leak near-identical poses and overstate performance. We also separated local audio/window dependencies from the browser demo.

## Accomplishments that we're proud of

We implemented an end-to-end prototype and report the recorded evaluation without substituting synthetic scores.

{table}

We trained on {m['train_size']} samples and evaluated on {m['test_size']} samples from the final numbered session. {split}

## What we learned

We learned to distinguish a working interface from evidence of generalization. We measure model-only latency separately from camera processing, UI and speech.

## Limitations and responsible use

We recorded {m['signers']} signer(s). Our small vocabulary and static single-hand features omit motion, facial expression and grammar. We store landmarks and labels, not camera images. Confidence is uncalibrated; this is not a replacement for human interpreters.

## What's next for SignSpeak

We would collect consented data from more signers, evaluate on an untouched signer/session, and study temporal and multi-hand features before making broader claims.
'''
    if len(story.split()) >= 600:
        raise SystemExit('STOP: story exceeds the requested word limit.')
    docs = ROOT / 'docs'
    docs.mkdir(exist_ok=True)
    (ROOT / 'README.md').write_text(readme)
    (docs / 'devpost_story.md').write_text(story)
    # Use a bundled font so thumbnail creation works on macOS and Linux.
    import matplotlib
    font_path = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
    font = lambda size: ImageFont.truetype(str(font_path), size)
    canvas = Image.new('RGB', (1500, 1000), '#0c1425')
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 40), 'SignSpeak', font=font(92), fill='#66e1c4')
    draw.text((75, 160), 'Sign language to text & speech', font=font(37), fill='white')
    draw.text((75, 225), f"Accuracy {m['accuracy']:.2%}   |   Macro F1 {m['macro_f1']:.4f}", font=font(34), fill='#b7c8e3')
    live = docs / 'screenshots/live_demo.png'
    with Image.open(live if live.exists() else matrix) as image:
        panel = ImageOps.contain(image.convert('RGB'), (1350, 620))
        canvas.paste(panel, ((1500-panel.width)//2, 320+(620-panel.height)//2))
    with Image.open(matrix) as image:
        inset = ImageOps.contain(image.convert('RGB'), (430, 330))
        canvas.paste(inset, (1000, 625))
    output = docs / 'thumbnail.png'
    canvas.save(output, optimize=True)
    if output.stat().st_size >= 5_000_000:
        raise SystemExit('STOP: thumbnail exceeds file size limit.')
    print('Generated README, Devpost story and thumbnail from recorded metrics.')


if __name__ == '__main__':
    main()
