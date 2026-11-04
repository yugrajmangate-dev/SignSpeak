## Inspiration

We are team YUGA—Yugraj Mangate and Gayatri Karanjkar. We wanted to explore a simple interface connecting visible hand gestures with text and speech, while being clear about the gap between gesture classification and full sign-language translation.

## What it does

Our Streamlit demo takes a camera snapshot, recognizes a supported static gesture, displays its label and confidence, and speaks editable text through the browser. Supported labels are Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou. We do not translate an alphabet or complete signed sentences.

## How we built it

We integrated Google's official pretrained MediaPipe Gesture Recognizer with Python, OpenCV, NumPy, Streamlit and browser Web Speech API. The bundled model works without recording training sessions. Our optional desktop path uses a majority vote, stable-token text and pyttsx3 speech. Codex and Claude assisted implementation and planning. Google MediaPipe supplied the pretrained model; we do not claim authorship of its weights.

## Challenges we ran into

Our planned custom-data workflow required recordings that we did not complete. We switched to an attributed pretrained gesture model rather than claiming unmeasured results. Cloud logs also exposed a Python compatibility mismatch; deployment must use the pinned project's supported Python version.

## Accomplishments that we're proud of

We connected an existing gesture model to a compact text-and-speech interface and made the supported vocabulary and evaluation status explicit.

| Field | Status |
| --- | --- |
| Accuracy | Not measured |
| Macro F1 | Not measured |
| Latency | Not measured |
| Evaluation samples | Not measured |
| Test session | Not measured |
| Evaluation signers | Not measured |

These fields are unmeasured in results/metrics.json. We have no project evaluation dataset, held-out test session or confusion matrix. Our sample inference checks do not establish accuracy or generalization.

## What we learned

We learned to separate model integration from training and validation. If we collect data later, we will split by session because random frame splits can leak nearly identical poses and inflate scores.

## Limitations and responsible use

We support a small set of static gestures, not full sign-language translation. The model omits language-specific motion and grammar. We performed no project-specific signer or cross-signer evaluation. The app does not store camera snapshots; optional collection stores landmarks and labels. It is not a replacement for human interpreters.

## What's next for SignSpeak

We would collect consented signer data, evaluate an untouched session, and investigate motion and language-specific vocabulary before making broader claims.
