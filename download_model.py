import tensorflow as tf
import os
import gdown

file_id = "1-UQCdlnIYo3EaqyA_urDpyGCysYXNSLS"
output = "models/model.keras"
os.makedirs("models", exist_ok=True)

# Download the actual model file from Google Drive
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False, fuzzy=True)

# Load the model with TensorFlow
try:
    model = tf.keras.models.load_model(output)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
