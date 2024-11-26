import os
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore 
from tensorflow.keras.datasets import mnist # type: ignore 
from tensorflow.keras.utils import to_categorical # type: ignore 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # For image processing

# Check if the model exists
model_path = 'mnist_digit_recognition_model.h5'

# Load the pre-trained model
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"{model_path} not found. Please train and save the model first.")

# Function to preprocess a custom image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img).astype('float32') / 255  # Normalize pixel values
    img_array = 1 - img_array  # Invert colors (MNIST digits are white on black)
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
    return img_array

# Predict on a custom image
def predict_custom_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Display the image and prediction
    plt.imshow(img_array.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_digit}, Confidence: {confidence:.2f}")
    plt.axis('off')
    plt.show()

    return predicted_digit, confidence

# image_directory = 'F:/books/FOUR YEAR/Data Mining/finial_project/'  # Replace with your directory
# for digit in range(10):  # Loop for digits 0 to 9
#     image_path = os.path.join(image_directory, f"{digit}.png")
    
#     if os.path.exists(image_path):
#         print(f"Processing {image_path}...")
#         predicted_digit, confidence = predict_custom_image(image_path)
#         print(f"Image: {image_path}, Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}\n")
#     else:
#         print(f"Image file {image_path} not found. Skipping...\n")
        
        

image_directory = 'F:/books/FOUR YEAR/Data Mining/finial_project/'  # Your image directory
image_name = 'test.png'  # Replace with the specific image name (e.g., '4.png')

image_path = os.path.join(image_directory, image_name)

if os.path.exists(image_path):
    print(f"Processing {image_path}...")
    predicted_digit, confidence = predict_custom_image(image_path)
    print(f"Image: {image_path}, Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}\n")
else:
    print(f"Image file {image_path} not found.")