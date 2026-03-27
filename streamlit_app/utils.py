"""
Age Detection Utilities
=======================
This module handles model loading, face detection, and age prediction.
"""

import os
import numpy as np
from PIL import Image
import cv2
import keras
from pathlib import Path

# Constants
MODEL_PATH = Path(__file__).parent / "models" / "Age_Sex_Detection.h5"
TARGET_SIZE = (224, 224)

# Global model instance
model = None


def load_model():
    """
    Load the Keras model for age and sex detection.
    """
    global model
    
    if model is None:
        print(f"Loading model from: {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
    
    return model


def get_face_detector():
    """
    Initialize and return OpenCV DNN face detector.
    Uses a pre-trained model from OpenCV.
    """
    # Use OpenCV's DNN module with a pre-trained face detection model
    # We'll use the Caffe model which doesn't require TensorFlow
    prototxt_path = "deploy.prototxt"
    caffemodel_path = "res10_300x300_ssd_iter_140000.caffemodel"
    
    # Check if model files exist, if not use Haar cascade as fallback
    detector = None
    if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
        detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    
    return detector


def detect_face_haar(image_array):
    """
    Detect face using Haar Cascade classifier.
    
    Args:
        image_array: numpy array of image in RGB format
        
    Returns:
        Cropped face image as numpy array, or None if no face detected
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        image_bgr,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None
    
    # Get the largest face
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, width, height = largest_face
    
    # Add padding around the face
    padding = int(max(width, height) * 0.2)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_bgr.shape[1], x + width + padding)
    y2 = min(image_bgr.shape[0], y + height + padding)
    
    # Crop the face
    face = image_bgr[y1:y2, x1:x2]
    
    # Convert back to RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    return face


def detect_face_dnn(image_array):
    """
    Detect face using OpenCV DNN with Caffe model.
    
    Args:
        image_array: numpy array of image in RGB format
        
    Returns:
        Cropped face image as numpy array, or None if no face detected
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    
    # Get detector
    detector = get_face_detector()
    if detector is None:
        return None
    
    # Detect faces
    detector.setInput(blob)
    detections = detector.forward()
    
    # Find the largest face with confidence > 0.5
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            faces.append((x1, y1, x2 - x1, y2 - y1))
    
    if len(faces) == 0:
        return None
    
    # Get the largest face
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, width, height = largest_face
    
    # Add padding around the face
    padding = int(max(width, height) * 0.2)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + width + padding)
    y2 = min(h, y + height + padding)
    
    # Crop the face
    face = image_bgr[y1:y2, x1:x2]
    
    # Convert back to RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    return face


def detect_face(image_array):
    """
    Detect and crop the largest face from an image.
    Tries DNN first, falls back to Haar cascade.
    
    Args:
        image_array: numpy array of image in RGB format
        
    Returns:
        Cropped face image as numpy array, or None if no face detected
    """
    # Try DNN first
    face = detect_face_dnn(image_array)
    if face is not None:
        return face
    
    # Fall back to Haar cascade
    face = detect_face_haar(image_array)
    return face


def preprocess_image(image):
    """
    Preprocess image for the model.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Preprocessed numpy array ready for model prediction
    """
    # Convert to RGB if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Detect and crop face
    face = detect_face(image)
    
    if face is None:
        # If no face detected, use the whole image resized
        face = cv2.resize(image, TARGET_SIZE)
    else:
        # Resize face to target size
        face = cv2.resize(face, TARGET_SIZE)
    
    # Normalize to [0, 1]
    face = face.astype('float32') / 255.0
    
    # Expand dimensions to match model input (1, 224, 224, 3)
    face = np.expand_dims(face, axis=0)
    
    return face


def predict_age_sex(image):
    """
    Predict age and sex from an image.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        Dictionary with 'age' (int) and 'sex' (string: 'Male' or 'Female')
    """
    load_model()
    
    # Preprocess image
    processed = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed, verbose=0)[0]
    
    # Assuming model outputs [age, sex_probability] or similar
    # Adjust based on actual model architecture
    age = int(round(predictions[0]))
    sex_prob = predictions[1] if len(predictions) > 1 else 0.5
    
    sex = "Male" if sex_prob > 0.5 else "Female"
    
    # Ensure age is in reasonable range
    age = max(1, min(100, age))
    
    return {
        "age": age,
        "sex": sex,
        "sex_confidence": float(abs(sex_prob - 0.5) * 2)  # Confidence as percentage
    }


def draw_prediction(image, age, sex):
    """
    Draw age and sex prediction on image.
    
    Args:
        image: PIL Image or numpy array
        age: Predicted age
        sex: Predicted sex
        
    Returns:
        Image with prediction drawn
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Create overlay
    overlay = image.copy()
    
    # Draw text background
    cv2.rectangle(overlay, (10, 10), (250, 80), (0, 100, 200), -1)
    
    # Add text
    cv2.putText(overlay, f"Age: {age}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, f"Sex: {sex}", (20, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Blend with original
    image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
    
    return image
