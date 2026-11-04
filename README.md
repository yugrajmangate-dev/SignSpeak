# SignSpeak — FT-04: Sign Language to Text/Speech Translator

Neural Storm 2026 · Track: Future Tech. A small Python prototype that maps isolated hand poses to a limited set of text tokens, with local text-to-speech. It does not translate full sign language.

## Team name and members

- Team: YUGA
- Members: Yugraj Mangate, Gayatri Karanjkar

## Problem

FT-04 asks for sign language to text/speech translation. Communication barriers motivate the project; this hackathon prototype explores only a small, explicitly collected vocabulary.

## Solution

Collect 21 MediaPipe landmarks per frame. Use only x/y, subtract the wrist, divide by maximum wrist-to-landmark distance, mirror left hands to right, then flatten to 42 values in `x0,y0,...,x20,y20` order. All camera paths mirror images before MediaPipe so handedness follows the same convention.

Train a RandomForest and a small MLP on earlier numbered sessions; reserve the highest numbered session for evaluation and winner selection by macro F1, breaking ties by accuracy. At least two sessions are required. No random frame split is used. The saved model retains the earlier-session fit; it is not retrained on the held-out session.

Desktop inference uses a full 10-frame window with a strict majority, then appends after one second of stability. A held sign appends once; remove your hand before repeating the same token. Letters concatenate; words get spaces. ENTER speaks, C clears, Q quits. Streamlit predicts from individual camera snapshots and displays model confidence; continuous inference runs locally, while the Streamlit sentence field can use browser speech.

## Pipeline

```mermaid
flowchart LR
    A[Webcam] --> B[MediaPipe 21 landmarks]
    B --> C[Normalization]
    C --> D[Classifier]
    D --> E[Majority vote]
    E --> F[Text/speech]
```

## Tech stack

Python 3.11, MediaPipe 0.10.14, OpenCV 4.10.0.84, NumPy, pandas, scikit-learn, joblib, matplotlib, pyttsx3, Streamlit. `pynput` tracks SPACE press/release locally. Direct dependency versions are pinned in the requirements files.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements-local.txt
# Ensure the final cv2 installation has desktop window support:
python -m pip install --force-reinstall --no-deps opencv-python==4.10.0.84
```

Grant camera access. On macOS, SPACE tracking also requires Input Monitoring/Accessibility permission for your terminal or IDE. On Linux, use a desktop X11 session for pynput and install `espeak` if speech initialization fails. Speech failure leaves text inference available.

`requirements.txt` is the cloud set and includes headless OpenCV. The pinned legacy MediaPipe also depends on `opencv-contrib-python`, which shares the `cv2` namespace; `packages.txt` supplies its Linux runtime libraries. The explicit local reinstall above selects GUI OpenCV for webcam windows.

## Run locally

```bash
python collect.py
# 0=A, 1=B, 2=C, 3=L, 4=V, 5=W, 6=Y, 7=OK, 8=THUMBS_UP, 9=HELLO
# Select a number while the OpenCV window is focused; hold SPACE; Q quits.
python collect.py  # new session: vary position, distance and lighting
python check_data.py
python train.py --signers 1
python infer.py
streamlit run app.py
```

Collect every intended class in training sessions and in the final session. Collect at least 100 frames per label in each session; vary poses during capture rather than holding an identical frame. A smaller vocabulary is easier to demonstrate in 60 minutes:

```bash
python collect.py --labels A B C
```

Keep the same label names across sessions. Each run creates the next numeric `data/session_<n>.csv` without overwriting a previous session. Empty sessions are rejected by training; remove an accidental empty session before retraining. Existing starter synthetic CSVs were preserved under `.smoke-test/legacy-data/` because their column headers were incompatible.

Verification:

```bash
python -m compileall -q common.py collect.py train.py infer.py app.py dry_run.py
python check_data.py
```

Recorded evaluation and submission assets are pending collection. After training, run `python prepare_submission.py` to generate documentation and the thumbnail from the saved metrics. Run `python presubmit_check.py` to list missing deliverables.

For Streamlit Community Cloud, push this project to a GitHub repository, including `app.py`, `common.py`, `requirements.txt`, `packages.txt`, `model.joblib`, and `results/`. Select **app.py** and **Python 3.11** in deployment settings. Do not upload `.venv` or personal capture data. Use only a model you trained or trust: joblib artifacts can execute code when loaded. Deployment is prepared but has not been published or cloud-tested. See [Streamlit dependency setup](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) and [deployment settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Screenshots

![Local demo](docs/screenshots/live_demo.png)
![Streamlit app](docs/screenshots/streamlit_app.png)
![Confusion matrix](results/confusion_matrix.png)

Images are pending recorded-data evaluation and real demo captures.

## Metrics

No recorded evaluation metrics are available yet. Collect sessions and train before reporting results.

Metrics and per-class precision/recall/F1/support are in `results/metrics.json`; the plot is `results/confusion_matrix.png`. Inference latency averages 50 warmed-up single-sample calls, excluding camera capture, landmark extraction, UI and speech. The last session also selects the winner, so these scores are not an independent final test. Use an additional untouched session for a stronger evaluation.

## AI tools used

AI tools used: Codex, Claude. Development and planning assistance was provided by these tools. No recorded performance is claimed yet.

## Limitations and responsible use

A small vocabulary, limited signer diversity, and correlated frames limit generalization to other people, environments and signing styles. Static single-hand x/y features omit motion, depth, two-hand interaction, facial expression and grammar. HELLO is only a held-pose shortcut. Unknown poses still receive a known label, and confidence is uncalibrated. This is not a replacement for human interpreters and should not be used for critical communication.

Camera snapshots are sent to the Streamlit server for in-memory processing; this app does not save them. Desktop collection intentionally saves landmark coordinates and labels locally, not webcam images. Webcam, keyboard permissions, speech hardware and actual sign recognition still require a hands-on check.
