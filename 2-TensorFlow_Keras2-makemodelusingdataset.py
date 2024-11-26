import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore 
from tensorflow.keras.utils import to_categorical # type: ignore 
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore 
from sklearn.model_selection import train_test_split
from PIL import Image

# Dataset path
dataset_path = "dataset/"  # Replace with your dataset folder path

# Preprocess the dataset
def load_data(dataset_path):
    images = []
    labels = []

    for label in range(10):  # Loop through folders 0-9
        label_path = os.path.join(dataset_path, str(label))
        if not os.path.exists(label_path):
            print(f"Folder {label} not found, skipping...")
            continue

        for file_name in os.listdir(label_path):
            file_path = os.path.join(label_path, file_name)
            try:
                # Open image and preprocess
                img = Image.open(file_path).convert('L')  # Convert to grayscale
                img = img.resize((28, 28))  # Resize to 28x28
                img_array = np.array(img).astype('float32') / 255.0  # Normalize
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
    
    images = np.array(images).reshape(-1, 28, 28, 1)  # Add channel dimension
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=10)  # One-hot encode labels

    return images, labels

# Load data
print("Loading data...")
X, y = load_data(dataset_path)
print(f"Loaded {X.shape[0]} images.")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')  # 10 classes (digits 0-9)
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Save the trained model
model.save("digit_recognizer.h5")
print("Model saved as 'digit_recognizer.h5'.")
