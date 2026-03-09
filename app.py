import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import subprocess
import sys

# install tensorflow dynamically (compatible version)
subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.17.0"])
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Voice Age Detection", layout="centered")

st.title("🎙️ Voice-based Age Detection (Male-only Logic)")
st.write("⚠️ This system processes **male voices only** as per internship task logic.")

# Load models once
@st.cache_resource
def load_models():
    gender_model = load_model("voice_gender.h5", compile=False)
    age_model = load_model("voice_age.h5", compile=False)
    return gender_model, age_model

voice_gender_model, voice_age_model = load_models()

# MFCC extractor
def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=3)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_len]

        mfcc = mfcc[..., np.newaxis]
        mfcc = np.expand_dims(mfcc, axis=0)
        return mfcc
    except Exception:
        return None

def detect_emotion_logic(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3)

        # Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

        # Energy
        rms = np.mean(librosa.feature.rms(y=y))

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # Simple rules
        if avg_pitch > 200 and rms > 0.04:
            return "Angry 😠"
        elif avg_pitch > 180 and rms > 0.02:
            return "Happy 😊"
        elif avg_pitch < 150 and rms < 0.02:
            return "Sad 😢"
        else:
            return "Neutral 😐"

    except:
        return "Unknown"
    
    # Age range converter
def age_to_range(age_value):
    if age_value < 20:
        return "Below 20"
    elif age_value < 30:
        return "20–30"
    elif age_value < 40:
        return "30–40"
    elif age_value < 60:
        return "40–60"
    else:
        return "60+"

# Gender prediction
def predict_voice_gender(audio_path):
    mfcc = extract_mfcc(audio_path)
    if mfcc is None:
        return "Unknown", 0.0

    pred = voice_gender_model.predict(mfcc, verbose=0)[0][0]
    gender = "Male" if pred >= 0.5 else "Female"
    confidence = float(pred if pred >= 0.5 else 1 - pred)
    return gender, round(confidence, 2)

# Age prediction
def predict_voice_age(audio_path):
    mfcc = extract_mfcc(audio_path)
    if mfcc is None:
        return "Unknown", None

    age_value = voice_age_model.predict(mfcc, verbose=0)[0][0]
    return age_to_range(age_value), round(float(age_value), 2)

# Upload UI
uploaded_audio = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(uploaded_audio.read())
        audio_path = temp.name

    st.audio(audio_path)

    with st.spinner("Analyzing voice..."):
        gender, gender_conf = predict_voice_gender(audio_path)

    st.subheader(f"Detected Gender: {gender} ({gender_conf})")

    if gender.lower() == "female":
        st.error("🚫 This task is restricted to male voices only.")
        os.remove(audio_path)
        st.stop()

    age_group, raw_age = predict_voice_age(audio_path)

    st.subheader(f"Predicted Age Group: {age_group}")
    st.write(f"Estimated Age Value: {raw_age}")

    if age_group == "60+":
       st.success("Senior citizen detected.")
       emotion = detect_emotion_logic(audio_path)
       st.subheader(f"Detected Emotion: {emotion}")
    else:
        st.info("Emotion analysis not required for non-senior age groups.")

    os.remove(audio_path)
