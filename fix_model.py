from tensorflow import keras

# Load your trained model
model = keras.models.load_model("cnnmodel.keras", compile=False)

# Save again in both formats
model.save("cnnmodel_fixed.keras")
model.save("cnnmodel_fixed.h5")

print("✅ Fixed models saved as cnnmodel_fixed.keras and cnnmodel_fixed.h5")
