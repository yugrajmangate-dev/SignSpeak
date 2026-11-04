"""Streamlit snapshot demo; desktop inference supplies continuous translation."""
import json
import streamlit.components.v1 as components
import cv2
import numpy as np
import streamlit as st
from common import ROOT
from gesture import create_recognizer, predict

st.set_page_config(page_title='SignSpeak', page_icon='🤟')
st.title('SignSpeak')
st.caption('Neural Storm 2026 · Future Tech · FT-04 · Pretrained gesture-to-text/speech prototype')
demo, performance, limitations = st.tabs(['Translator', 'Model performance', 'Limitations & responsible use'])
with demo:
    predicted = ''
    st.info('Pretrained MediaPipe gesture demo. No custom training or project accuracy evaluation has been performed.')
    st.caption('Supported poses: closed fist, open palm, pointing up, thumbs down, thumbs up, victory, I love you. These are gesture labels, not full sign-language translation.')
    image = st.camera_input('Take a photo of one hand')
    if image:
        frame = cv2.imdecode(np.frombuffer(image.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error('Could not decode image. Please retake it.')
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with create_recognizer() as recognizer:
                label, confidence = predict(recognizer, rgb)
            if label:
                predicted = 'I love you' if label == 'ILoveYou' else label.replace('_', ' ')
                st.subheader(predicted)
                st.progress(confidence, text=f'Model confidence: {confidence:.1%} (uncalibrated)')
            else:
                st.info('No supported gesture recognized. Show your whole hand clearly and retake the photo.')
    sentence = st.text_area('Sentence to speak', value=predicted.replace('_', ' '))
    speech_text = json.dumps(sentence).replace('<', '\u003c')
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
    st.write('This demo uses an official pretrained MediaPipe model without project-specific training or evaluation. It recognizes a small set of static gestures, not an alphabet or full sign language. It does not handle motion, facial expression or grammar and is not a replacement for human interpreters.')
    st.write('Snapshots are sent to the Streamlit server for in-memory processing and are not saved by this app. Confidence is uncalibrated. No accuracy, macro F1 or signer-generalization claim is made.')
    st.markdown('[Official model documentation](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer)')
