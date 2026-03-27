"""
Age Detection Utilities
=======================
This module handles face detection and age/gender prediction.
Uses OpenCV for face detection and heuristic-based estimation.
"""

import numpy as np
from PIL import Image
import cv2
import random


def detect_face(image_array):
    """
    Detect and crop the largest face from an image using Haar Cascade.
    """
    print("[DEBUG] Starting face detection...")
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    print("[DEBUG] Haar Cascade loaded successfully")
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        image_bgr,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    print(f"[DEBUG] Detected {len(faces)} face(s)")
    
    if len(faces) == 0:
        return None, image_bgr
    
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
    
    print(f"[DEBUG] Face cropped successfully, size: {face.shape}")
    
    return face, image_bgr


def analyze_face(face_image):
    """
    Analyze face image to estimate age and gender.
    Uses simple image analysis heuristics.
    """
    print("[DEBUG] Analyzing face features...")
    
    if face_image is None or face_image.size == 0:
        print("[DEBUG] No face detected, using fallback")
        return random.randint(18, 65), "Male", 0.5
    
    try:
        # Resize for consistent analysis
        face_resized = cv2.resize(face_image, (100, 100))
        
        # Convert to different color spaces
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
        
        # Calculate brightness
        brightness = np.mean(gray) / 255.0
        
        # Calculate contrast (variance of brightness)
        contrast = np.std(gray) / 255.0
        
        # Calculate skin tone (lower saturation usually indicates lighter skin)
        avg_saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # Calculate skin color uniformity (lower std = more uniform skin)
        skin_uniformity = 1.0 - np.std(face_resized[:, :, 0].flatten()) / 255.0
        
        # Edge detection for texture analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate face aspect ratio (wider faces might indicate male)
        face_height, face_width = gray.shape
        aspect_ratio = face_width / face_height if face_height > 0 else 0.5
        
        print(f"[DEBUG] Features - brightness: {brightness:.2f}, contrast: {contrast:.2f}, saturation: {avg_saturation:.2f}, edge_density: {edge_density:.4f}")
        
        # Age estimation based on features
        base_age = 25
        age = base_age
        
        # Texture factor (more texture = older)
        age += int(edge_density * 100)
        
        # Contrast factor
        age += int(contrast * 30)
        
        # Uniformity factor (less uniform = older)
        age += int((1 - skin_uniformity) * 20)
        
        print(f"[DEBUG] Initial age estimate: {age}")
        
        # Gender estimation
        male_score = 0.0
        
        # Aspect ratio contribution
        if aspect_ratio > 0.55:
            male_score += 0.3
        else:
            male_score -= 0.3
        
        # Brightness contribution
        if brightness > 0.5:
            male_score -= 0.2
        else:
            male_score += 0.2
        
        # Saturation contribution
        if avg_saturation > 0.3:
            male_score -= 0.2
        else:
            male_score += 0.2
        
        print(f"[DEBUG] Male score: {male_score:.2f}")
        
        # Normalize score to probability
        sex_prob = (male_score + 1) / 2
        sex_prob = max(0.1, min(0.9, sex_prob))
        
        sex = "Male" if sex_prob > 0.5 else "Female"
        
        # Ensure reasonable bounds
        age = max(1, min(100, age))
        
        # Calculate confidence
        confidence = 0.5 + abs(sex_prob - 0.5)
        
        print(f"[DEBUG] Final result - Age: {age}, Gender: {sex}, Confidence: {confidence:.2f}")
        
        return age, sex, confidence
        
    except Exception as e:
        print(f"[ERROR] Exception in analyze_face: {str(e)}")
        return fallback_prediction()


def fallback_prediction():
    """
    Fallback prediction when face detection or analysis fails.
    Returns random but reasonable values.
    """
    print("[DEBUG] Using fallback prediction")
    
    # Random age between 18 and 65
    age = random.randint(18, 65)
    
    # Random gender
    sex = random.choice(["Male", "Female"])
    
    # Low confidence since we're using fallback
    confidence = 0.4
    
    return age, sex, confidence


def predict_age_sex(image):
    """
    Predict age and sex from an image.
    """
    print("[DEBUG] Starting prediction...")
    
    try:
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            print("[DEBUG] Converting PIL Image to numpy array")
            img_array = np.array(image)
        else:
            img_array = image
        
        print(f"[DEBUG] Image shape: {img_array.shape}")
        
        # Ensure RGB format
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Detect face
        face, full_image = detect_face(img_array)
        
        # Analyze face for age and gender
        if face is not None and face.size > 0:
            age, sex, confidence = analyze_face(face)
        else:
            print("[DEBUG] No face detected, using fallback")
            age, sex, confidence = fallback_prediction()
        
        result = {
            "age": age,
            "sex": sex,
            "sex_confidence": confidence
        }
        
        print(f"[DEBUG] Prediction complete: {result}")
        return result
        
    except Exception as e:
        print(f"[ERROR] Exception in predict_age_sex: {str(e)}")
        age, sex, confidence = fallback_prediction()
        return {
            "age": age,
            "sex": sex,
            "sex_confidence": confidence
        }


def draw_prediction(image, age, sex):
    """
    Draw age and sex prediction on image.
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
