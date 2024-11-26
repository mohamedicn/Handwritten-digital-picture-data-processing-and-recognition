from tensorflow.keras.models import load_model # type: ignore 
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("digit_recognizer.h5")
print("Model loaded successfully.")

# Preprocess the image for prediction
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img).astype('float32') / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model
    return img_array

# Predict the digit in the image
def predict_digit(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Display the processed image
    img = img_array.reshape(28, 28)  # Remove batch and channel dimensions for display
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {predicted_digit}, Confidence: {confidence:.2f}")
    plt.axis('off')
    plt.show()
    
    return predicted_digit, confidence

# Path to the images directory
image_directory = 'F:/books/FOUR YEAR/Data Mining/finial_project/'  # Replace with your directory

# Loop through images named 0.png to 9.png
for digit in range(10):  # Loop for digits 0 to 9
    image_path = os.path.join(image_directory, f"{digit}.png")
    
    if os.path.exists(image_path):
        print(f"Processing {image_path}...")
        predicted_digit, confidence = predict_digit(image_path)
        print(f"Image: {image_path}, Predicted Digit: {predicted_digit}, Confidence: {confidence:.2f}\n")
    else:
        print(f"Image file {image_path} not found. Skipping...\n")
