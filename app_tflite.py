from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from random import random

app = Flask(__name__)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open('lb.pickle', 'rb') as f:
    lb = pickle.load(f)

def is_valid_image(image):
    """Simple sanity checks to reject invalid images"""
    if np.mean(image) < 30:
        return False, "Image too dark"
    
    
    if np.mean(image) > 225:
        return False, "Image too bright"
    
    # Check if image has very low variance (solid color)
    if np.var(image) < 100:
        return False, "Image appears to be solid color"
    
    return True, "Valid image"

def model_predict(img_path):
    """Predict the disease class for a single image"""
    if not os.path.exists(img_path):
        return "Error: Image file not found!", 0.0
    
    
    image = cv2.imread(img_path)
    if image is None:
        return "Error: Could not load image!", 0.0
    
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform sanity checks
    is_valid, error_msg = is_valid_image(image)
    if not is_valid:
        return f"Invalid image: {error_msg}", 0.0
    
    # Resize to 224x224 (ResNet50 input size)
    image = cv2.resize(image, (224, 224))
    
    
    image = image.astype("float32")
    
    image = np.expand_dims(image, axis=0)
    
    # Make prediction using TFLite
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # Get the predicted class
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    predicted_class = lb.classes_[class_idx]
    
    return predicted_class, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('static', 'last_uploaded.jpg')
            file.save(filepath)
            label, confidence = model_predict(filepath)
           
            print(f"Raw confidence: {confidence}")
            
            confidence_percent = min(max(confidence, 0.0), 1.0) * 100
            print(f"Confidence percent: {confidence_percent}")
            return render_template('predict.html', label=label, confidence=round(confidence_percent, 2), random=random)
    return render_template('predict.html')

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
