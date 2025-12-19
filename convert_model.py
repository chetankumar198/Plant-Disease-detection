import tensorflow as tf
from tensorflow import keras

# Load old model using legacy loader
old_model = keras.models.load_model(
    "CNN_plantdiseases_model.keras",
    compile=False,
    safe_mode=False
)

# Rebuild model properly (forces new Input layer)
new_model = keras.Sequential()
new_model.add(keras.Input(shape=(224, 224, 3)))

for layer in old_model.layers:
    if not isinstance(layer, keras.layers.InputLayer):
        new_model.add(layer)

# Save in true Keras 3 format
new_model.save("CNN_plantdiseases_model_keras3.keras")

print("✅ Model converted successfully")
