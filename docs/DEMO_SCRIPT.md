# SignSpeak screen-recording script

Target duration: 50 seconds. Timing is a filming plan, not a measured system result.

- 0–8 seconds: Show the app title. Say: “We are YUGA. SignSpeak is a pretrained static gesture-to-text and speech prototype.”
- 8–20 seconds: Hold Thumb_Up, take a photo, and show the actual displayed label and confidence. Say: “This uses Google's pretrained MediaPipe model. We did not train these weights.” Do not quote a confidence before observing it.
- 20–30 seconds: Try Open_Palm or Victory. Show whichever result actually appears. If recognition fails, show the failure rather than describing a success.
- 30–40 seconds: Show the editable text and click Speak sentence. Capture actual browser audio if available; otherwise state that audio was unavailable.
- 40–50 seconds: Open Model performance. Say: “Project accuracy and F1 have not been evaluated. This recognizes limited static gestures, not full sign language, and is not a replacement for interpreters.”

Use only the supported labels listed in results/metrics.json. Do not claim an accuracy, latency, dataset size, held-out session or cross-signer result. Remove notifications and personal information from the recording.
