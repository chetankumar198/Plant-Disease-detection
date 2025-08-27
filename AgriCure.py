import streamlit as st
from tornado.websocket import WebSocketClosedError
import numpy as np
print(np.__version__)
import cv2
print(cv2.__version__)
import os
from PIL import Image
import gdown
import h5py
import tensorflow as tf
import time
import logging

# ⚡ Updated imports for model loading
from tensorflow import keras  

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

# ------------------ Download & Load Model ------------------
@st.cache_resource
def download_and_load_model(file_id, local_path="models/CNN_plantdiseases_model.keras", max_retries=3):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"

    # Download with retry
    if not os.path.exists(local_path):
        for attempt in range(max_retries):
            try:
                st.info(f"Downloading model... Attempt {attempt + 1}")
                gdown.download(url, local_path, quiet=False, fuzzy=True)
                if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
                    st.success("✅ Model downloaded successfully!")
                    break
            except Exception as e:
                st.warning(f"Download attempt {attempt+1} failed: {e}")
                time.sleep(2)
        else:
            st.error("❌ Failed to download the model after multiple attempts.")
            return None

    # ✅ Load the modern .keras model directly
    try:
        model = keras.models.load_model(local_path, compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Google Drive file ID
FILE_ID = "1-UQCdlnIYo3EaqyA_urDpyGCysYXNSLS"
model = download_and_load_model(FILE_ID)

# ------------------ Prediction Function ------------------
def model_predict(image):
    if model is None:
        st.error("Model could not be loaded.")
        return None, None

    try:
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            st.error("Unsupported image format.")
            return None, None

        if img is None:
            st.error("Error: Image not found or invalid format.")
            return None, None

        H, W, C = 224, 224, 3
        img = cv2.resize(img, (H, W))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = img.reshape(1, H, W, C)

        prediction = model.predict(img)[0]
        result_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        return result_index, confidence
    except Exception as e:
        st.error(f"Error during image processing or prediction: {e}")
        return None, None

# ------------------ CSS Styling ------------------
st.markdown("""
<style>
body { background-color: #a8d8a8; color: #000; }
.stSidebar { background: linear-gradient(to right, #7cb77c, #a8d8a8); box-shadow: 0 4px 8px rgba(0,0,0,0.2); padding:10px; border-radius:15px;}
.stSidebar img { box-shadow:0 6px 12px rgba(0,0,0,0.3), 0 0 10px rgba(124,183,124,0.5); border-radius:10px; }
.stButton>button { background: linear-gradient(to bottom, #7cb77c, #a8d8a8); color:white; border:none; border-radius:10px; padding:10px 20px; transition: transform 0.2s ease, box-shadow 0.2s ease; }
.stButton>button:hover { transform: scale(1.05); box-shadow:0 4px 8px rgba(0,0,0,0.2); }
.content-box { background:white; border-radius:15px; box-shadow:0 6px 12px rgba(0,0,0,0.3); padding:20px; }
footer { position: fixed; bottom:0; width:100%; text-align:center; background-color:#7cb77c; color:white; padding:10px; }
</style>
""", unsafe_allow_html=True)

# ------------------ Sidebar ------------------
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# ✅ Safe image loading
try:
    sidebar_img = Image.open("assets/farm_sunset.webp")
    st.sidebar.image(sidebar_img, caption="Healthy Crops")
except FileNotFoundError:
    st.sidebar.warning("⚠️ Sidebar image not found. Please add it to /assets folder.")

# ------------------ Class Names ------------------
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ------------------ Main App ------------------
if app_mode == "HOME":
    st.markdown("""
    <div class="content-box">
        <h1 style='text-align: center;'>🌿 Plant Disease Detection System 🌱</h1>
        <p style='text-align: center;'>Empowering farmers with AI-driven solutions.</p>
    </div>
    """, unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.markdown("""
    <div class="content-box">
        <h2>Plant Disease Detection</h2>
    </div>
    """, unsafe_allow_html=True)

    option = st.radio("Choose an option:", ["Upload an Image", "Scan via Webcam"])

    if option == "Upload an Image":
        test_image = st.file_uploader("Choose an Image:")
        if test_image is not None:
            img = Image.open(test_image)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            if st.button("Predict"):
                try:
                    result_index, confidence = model_predict(img)
                    if result_index is not None and result_index < len(class_name):
                        st.success(f"Predicted class: {class_name[result_index]} (Confidence: {confidence:.2%})")
                    else:
                        st.error("Invalid prediction index or no prediction made.")
                except WebSocketClosedError:
                    st.warning("⚠️ Connection closed while predicting. Please try again.")

    elif option == "Scan via Webcam":
        camera_image = st.camera_input("Capture Image")
        if camera_image is not None:
            img = Image.open(camera_image)
            st.image(img, caption="Captured Image", use_column_width=True)
            if st.button("Predict"):
                try:
                    result_index, confidence = model_predict(img)
                    if result_index is not None and result_index < len(class_name):
                        st.success(f"Predicted class: {class_name[result_index]} (Confidence: {confidence:.2%})")
                    else:
                        st.error("Invalid prediction index or no prediction made.")
                except WebSocketClosedError:
                    st.warning("⚠️ Connection closed while predicting. Please try again.")

# ------------------ Footer ------------------
st.markdown("""
<footer>
    <p>🌱 Empowering farmers, one crop at a time.</p>
</footer>
""", unsafe_allow_html=True)
