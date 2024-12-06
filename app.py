# -------------
# Version-4
# -------------

import os
from flask import Flask, render_template, request, send_from_directory
from keras_preprocessing import image
import numpy as np
import tensorflow as tf
import cv2  # Import OpenCV for face detection


app = Flask(__name__)

# Define paths for static assets and model folders
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
MODEL_FOLDER = os.path.join(STATIC_FOLDER, 'models')

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Global variable for the model
model = None


def load_model():
    """Load model once at running time for all predictions"""
    global model
    print('[INFO] : Model loading ................')
    model = tf.keras.models.load_model(os.path.join(MODEL_FOLDER, 'cat_dog_classifier.h5'))
    print('[INFO] : Model loaded')


def detect_human_face(image_path):
    """Detects if a human face is present in the image with stricter filtering."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces with even stricter parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,  # Increase further to reduce false positives
        minSize=(120, 120)  # Larger minSize to avoid detecting cat faces
    )

    # Filter detections based on aspect ratio and print for debugging
    for (x, y, w, h) in faces:
        aspect_ratio = h / w  # Calculate height-to-width ratio
        if 0.8 < aspect_ratio < 1.2:  # Typical human face aspect ratio
            print(f"Detected potential human face at {(x, y, w, h)} with aspect ratio: {aspect_ratio}")
            return True  # Return True if a human-like face is detected

    # If no valid human faces are found, return False
    return False

def predict(fullpath):
    # Check if a human face is detected in the image
    if detect_human_face(fullpath):
        return 'Human', 0  # If a face is detected, return "Unknown" with 0% confidence

    # Proceed with cat/dog classification if no face is detected
    data = image.load_img(fullpath, target_size=(128, 128, 3))
    data = np.expand_dims(data, axis=0)
    data = data.astype('float') / 255  # Normalize pixel values

    # Prediction without using `graph.as_default()` since eager mode is enabled
    result = model.predict(data)
    pred_prob = result.item()

    # Use the CNN model's probabilities for single-class predictions
    if 0.4 <= pred_prob <= 0.62:
        print(f"[DEBUG] TensorFlow model probability: {pred_prob}")            
        label = 'Uncertain'
        accuracy = 0  # Uncertain predictions have no meaningful confidence

    elif pred_prob > 0.62:
        print(f"[DEBUG] TensorFlow model probability: {pred_prob}")
        label = 'Dog'
        accuracy = round(pred_prob * 100, 2)
    else:
        print(f"[DEBUG] TensorFlow model probability: {pred_prob}")
        label = 'Cat'
        accuracy = round((1 - pred_prob) * 100, 2)

    return label, accuracy


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Save the uploaded file
    file = request.files['image']
    fullname = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(fullname)

    # Predict the label
    label, accuracy = predict(fullname)

    return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)

@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/about')
def about():
    return render_template('about.html')

def create_app():
    load_model()
    return app

# if __name__ == '__main__':
#     app = create_app()
#     app.run(debug=True)

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)