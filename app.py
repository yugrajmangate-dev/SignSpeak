"""Streamlit snapshot demo; desktop inference supplies continuous translation."""
import json
import streamlit.components.v1 as components
import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from common import ROOT, normalize

st.set_page_config(page_title='SignSpeak', page_icon='🤟')
st.title('SignSpeak')
st.caption('Neural Storm 2026 · Future Tech · FT-04 · Small-vocabulary hand-pose prototype')
demo, performance, limitations = st.tabs(['Translator', 'Model performance', 'Limitations & responsible use'])
with demo:
    predicted = ''
    path = ROOT / 'model.joblib'
    if not path.exists():
        st.info('Collect at least two real sessions and run train.py, then include model.joblib in deployment.')
    else:
        bundle = joblib.load(path)
        if bundle.get('synthetic'):
            st.warning('Synthetic smoke-test model only. No real sign recognition demonstrated.')
        image = st.camera_input('Take a photo of one hand')
        if image:
            frame = cv2.imdecode(np.frombuffer(image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                st.error('Could not decode image. Please retake it.')
            else:
                # MediaPipe handedness assumes mirrored/selfie input, as in collect.py.
                rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
                    result = hands.process(rgb)
                if not result.multi_hand_landmarks:
                    st.info('No hand detected. Retake with your whole hand visible.')
                else:
                    features = normalize(result.multi_hand_landmarks[0].landmark, result.multi_handedness[0].classification[0].label)
                    if features is not None:
                        model = bundle['model']
                        probabilities = model.predict_proba(features.reshape(1, -1))[0]
                        index = int(probabilities.argmax())
                        predicted = str(model.classes_[index])
                        st.subheader(predicted)
                        st.progress(float(probabilities[index]), text=f'Model confidence: {probabilities[index]:.1%} (uncalibrated)')
    sentence = st.text_area('Sentence to speak', value=predicted.replace('_', ' '))
    speech_text = json.dumps(sentence).replace('<', '\\u003c')
    components.html(f"""
        <button id="speak" type="button">Speak sentence</button>
        <span id="status" role="status"></span>
        <script>
        document.getElementById('speak').onclick = () => {{
            const status = document.getElementById('status');
            if (!('speechSynthesis' in window)) {{
                status.textContent = 'Speech is unavailable in this browser.';
                return;
            }}
            const text = {speech_text};
            if (!text.trim()) {{ status.textContent = 'Enter a sentence first.'; return; }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.onerror = () => {{ status.textContent = 'Browser speech failed.'; }};
            window.speechSynthesis.speak(utterance);
        }};
        </script>
    """, height=60)
with performance:
    metrics = ROOT / 'results' / 'metrics.json'
    if metrics.exists():
        st.json(json.loads(metrics.read_text()))
    else:
        st.info('No training results yet.')
    matrix = ROOT / 'results' / 'confusion_matrix.png'
    if matrix.exists():
        st.image(str(matrix), caption='Held-out last session')
with limitations:
    st.write('Single-signer data and a small vocabulary limit generalization. Static hand poses do not capture motion, facial expression, or sign-language grammar. HELLO is a held-pose shortcut, not dynamic sign recognition. This prototype is not a replacement for human interpreters.')
    st.write('Camera snapshots are processed on the Streamlit server in memory and are not saved by this app. Model confidence is not a guarantee of correctness.')
