# Wheat Leaf Disease Detector 🌾

This project is a Machine Learning web application designed to automatically detect and classify diseases in wheat leaves from uploaded images. It uses a deep learning model (ResNet50 architecture) that has been trained to identify common wheat diseases with high accuracy.

## 🎯 Supported Classifications
The model is trained to detect the following classes:
*   **Fusarium Head Blight**
*   **Healthy Wheat**
*   **Leaf Rust**
*   **Tan Spot**
*   **Unknown** (If the image doesn't resemble a wheat leaf)

## 🛠️ Tech Stack
*   **Backend:** Python, Flask
*   **Machine Learning:** TensorFlow, Keras, TensorFlow Lite
*   **Image Processing:** OpenCV, NumPy
*   **Deployment:** Docker, Gunicorn

## 🚀 Features
*   **Web Interface:** A simple, user-friendly Flask interface that allows users to upload leaf images and instantly view predictions.
*   **TFLite Optimization:** The original 223MB Keras model (`.h5`) was optimized and compressed into a lightweight 25MB TensorFlow Lite model (`.tflite`). This allows the application to run smoothly on free-tier cloud environments with limited memory (like Render.com) without sacrificing accuracy.

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rohit6863/Wheat-Leaf-Disease-Detector.git
   cd Wheat-Leaf-Disease-Detector
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application:**
   ```bash
   # Run the optimized TFLite version
   python app_tflite.py
   ```

5. **Open your browser:**
   Navigate to `http://127.0.0.1:5000` to use the web application.

## 🐳 Deployment (Docker)
The project includes a `Dockerfile` pre-configured for platforms like Render or Koyeb. It uses Gunicorn to serve the Flask app (`app_tflite.py`) and reads the `$PORT` environment variable assigned by the host.

## 📊 Model Performance
The original model was trained over 12 epochs using a frozen ResNet50 backbone, followed by fine-tuning. 
Based on the testing data, the model achieved an overall **F1-score of ~96%**, showing exceptional precision across all classes.

---
*Created as part of a Project Based Learning (PBL) curriculum.*
