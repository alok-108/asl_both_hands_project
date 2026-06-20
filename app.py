import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import collections
from statistics import mode

# Try to import tensorflow for TFLite
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Configuration
MODEL_PATH = "asl_hand_landmark_model.tflite"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SMOOTHING_FRAMES = 5  # Number of frames for temporal smoothing

# Temporal Smoothing History (Tracked by Handedness)
history = {
    "Left": collections.deque(maxlen=SMOOTHING_FRAMES),
    "Right": collections.deque(maxlen=SMOOTHING_FRAMES)
}

# Manual drawing connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

@st.cache_resource
def load_detector():
    """Loads MediaPipe HandLandmarker"""
    import urllib.request
    if not os.path.exists('hand_landmarker.task'):
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 
            'hand_landmarker.task'
        )
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)

@st.cache_resource
def load_tflite_model():
    """Loads the TFLite model and allocates tensors."""
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow is not installed. Please install it to use TFLite.")
        st.stop()
        
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model '{MODEL_PATH}' not found! Please run `python train_model.py` to generate it.")
        st.stop()
        
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return interpreter, input_details, output_details
    except Exception as e:
        st.error(f"❌ Failed to load TFLite model: {e}")
        st.stop()

def process_landmarks(hand_landmarks, handedness):
    """Normalize and format landmarks to a 63-element array (x, y, z)"""
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    wrist_z = hand_landmarks[0].z
    
    shifted = []
    for lm in hand_landmarks:
        shifted.append([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
    shifted = np.array(shifted)
    
    # Scale by maximum Euclidean distance from wrist
    distances = np.linalg.norm(shifted, axis=1)
    max_dist = np.max(distances)
    if max_dist > 0:
        shifted = shifted / max_dist
        
    # Mirror x-coordinates for Left hand to make it hand-agnostic
    if handedness == "Left":
        shifted[:, 0] = -shifted[:, 0]
        
    return shifted.flatten().astype(np.float32)

def draw_landmarks_and_box(image, landmarks, label):
    height, width, _ = image.shape
    pixel_landmarks = []
    
    for lm in landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        pixel_landmarks.append((x, y))
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks):
            cv2.line(image, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 255, 0), 2)
            
    x_coords = [p[0] for p in pixel_landmarks]
    y_coords = [p[1] for p in pixel_landmarks]
    x1, y1 = max(0, min(x_coords) - 20), max(0, min(y_coords) - 20)
    x2, y2 = min(width, max(x_coords) + 20), min(height, max(y_coords) + 20)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

def get_smoothed_prediction(handedness, prediction_idx, confidence):
    """Applies temporal smoothing (mode filter) over the last N frames."""
    history[handedness].append(prediction_idx)
    
    try:
        # Find the most frequent prediction in recent frames
        smoothed_idx = mode(history[handedness])
    except:
        # If there's a tie, fallback to the latest
        smoothed_idx = prediction_idx
        
    predicted_letter = CLASSES[smoothed_idx]
    return f"{handedness}: {predicted_letter} ({confidence:.0f}%)"

def main():
    st.set_page_config(page_title="ASL Both Hands", layout="wide")
    st.title("Real-Time ASL Alphabet Recognition")
    
    # Initialize Core Components
    detector = load_detector()
    interpreter, input_details, output_details = load_tflite_model()
    
    st.sidebar.title("Model Information")
    st.sidebar.success("✅ Powered by Production-Grade TFLite Model")
    st.sidebar.write("- **Architecture:** 63-Feature (x, y, z depth)")
    st.sidebar.write("- **Hand-Agnostic:** Yes")
    st.sidebar.write(f"- **Temporal Smoothing:** {SMOOTHING_FRAMES} frames")
    st.sidebar.write("---")
    
    run = st.checkbox('Start Webcam', key='run_webcam')
    FRAME_WINDOW = st.empty()
    
    # Clear history when webcam is toggled to prevent artifacting from old sessions
    if not run:
        history["Left"].clear()
        history["Right"].clear()
        st.write("Webcam is stopped.")
        return

    cap = cv2.VideoCapture(0)
    
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break
            
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        detection_result = detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                handedness = detection_result.handedness[idx][0].category_name # "Left" or "Right"
                
                # Extract 63-element feature vector (x, y, z)
                features = process_landmarks(hand_landmarks, handedness)
                
                # Run TFLite Inference
                interpreter.set_tensor(input_details[0]['index'], np.array([features]))
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0]
                
                # Extract Confidence Score
                class_idx = np.argmax(prediction)
                confidence = prediction[class_idx] * 100
                
                # Temporal Smoothing
                label_text = get_smoothed_prediction(handedness, class_idx, confidence)
                
                # Draw
                draw_landmarks_and_box(frame_rgb, hand_landmarks, label_text)
                
        FRAME_WINDOW.image(frame_rgb)
        
    cap.release()

if __name__ == "__main__":
    main()
