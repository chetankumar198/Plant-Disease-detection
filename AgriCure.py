import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import gdown
import tensorflow as tf
from tensorflow import keras
import logging

# ------------------ BASIC CONFIG ------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide"
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel(logging.ERROR)

# ------------------ MODEL DOWNLOAD & LOAD ------------------
@st.cache_resource
def load_model():
    MODEL_PATH = "models/CNN_plantdiseases_model.keras"

    FILE_ID = "YOUR_REAL_GOOGLE_DRIVE_FILE_ID"  # ✅ replace this

    os.makedirs("models", exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# ------------------ CLASS NAMES ------------------
CLASS_NAMES = [
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
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ------------------ PREDICTION FUNCTION ------------------
def predict(image: Image.Image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    return CLASS_NAMES[idx], confidence

# ------------------ SIDEBAR ------------------
st.sidebar.title("🌱 Plant Disease Detection")
page = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

try:
    st.sidebar.image("assets/farm_sunset.webp", caption="Healthy Crops")
except:
    pass

# ------------------ HOME PAGE ------------------
if page == "HOME":
    st.markdown(
        """
        <h1 style="text-align:center;">🌿 Plant Disease Detection System</h1>
        <p style="text-align:center;">
        AI-powered crop disease identification to help farmers take early action.
        </p>
        """,
        unsafe_allow_html=True
    )

# ------------------ DISEASE RECOGNITION ------------------
if page == "DISEASE RECOGNITION":
    st.header("🌾 Disease Recognition")

    uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            with st.spinner("🔍 Analyzing image..."):
                label, conf = predict(image)

            st.success(f"**Predicted Disease:** {label}")
            st.info(f"**Confidence:** {conf:.2%}")

# ------------------ FOOTER ------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center;">
    🌱 Empowering farmers, one crop at a time.
    </p>
    """,
    unsafe_allow_html=True
)
