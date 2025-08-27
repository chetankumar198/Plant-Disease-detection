import streamlit as st
import tensorflow as tf
import os
import gdown

st.title("Model Loader with Streamlit")

file_id = "1-UQCdlnIYo3EaqyA_urDpyGCysYXNSLS"
output = "models/model.keras"
os.makedirs("models", exist_ok=True)

with st.spinner("Downloading model..."):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False, fuzzy=True)

try:
    model = tf.keras.models.load_model(output)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
