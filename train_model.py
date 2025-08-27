from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image size to 224x224x3 (RGB)
img_size = (224, 224)

# Data preprocessing (change 'dataset/train' and 'dataset/test' to your actual folders)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=img_size,
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=img_size,
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(train_generator, validation_data=val_generator, epochs=5)

# Save in .keras format
model.save("models/cnnmodel_fixed.keras")
print("✅ Plant Disease Model saved as cnnmodel_fixed.keras")
