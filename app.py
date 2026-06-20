import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Try to import tensorflow for local fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Sequential
    from tensorflow.keras.layers import Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Configuration
MODEL_PATH = "asl_hand_landmark_model.h5"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

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

class DummyModel:
    def predict(self, features):
        return [np.random.rand(26)]

@st.cache_resource
def load_asl_model():
    """
    Attempts to load models in this priority:
    1. smart_gestures (PyPI package)
    2. Local asl_hand_landmark_model.h5
    3. Dummy Model (fallback)
    """
    status_msg = ""
    model_source = "dummy"
    
    # 1. Try smart_gestures
    try:
        from smart_gestures import ASLModel
        model = ASLModel.load_pretrained()
        model_source = "smart_gestures"
        return model, model_source, "Successfully loaded `smart_gestures` pre-trained model!"
    except ImportError:
        status_msg += "Could not import `smart_gestures`. "
    except Exception as e:
        status_msg += f"Failed to load `smart_gestures` model: {e}. "
        
    # 2. Try Local .h5 Model
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            model_source = "local_h5"
            return model, model_source, status_msg + f"Successfully loaded local `{MODEL_PATH}`!"
        except Exception as e:
            status_msg += f"Failed to load local .h5 model: {e}. "
    elif not TF_AVAILABLE:
        status_msg += "Tensorflow is not installed, cannot load local .h5 model. "
    else:
        status_msg += f"Local `{MODEL_PATH}` not found. "
        
    # 3. Fallback to Dummy Model
    if TF_AVAILABLE:
        model = Sequential([
            Dense(128, activation='relu', input_shape=(42,)),
            Dense(26, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    else:
        model = DummyModel()
        
    status_msg += "Falling back to a DUMMY model with random predictions."
    return model, "dummy", status_msg

def process_landmarks(hand_landmarks, handedness):
    """Normalize and format landmarks to a 42-element array"""
    wrist_x = hand_landmarks[0].x
    wrist_y = hand_landmarks[0].y
    
    shifted = []
    for lm in hand_landmarks:
        shifted.append([lm.x - wrist_x, lm.y - wrist_y])
    shifted = np.array(shifted)
    
    distances = np.linalg.norm(shifted, axis=1)
    max_dist = np.max(distances)
    if max_dist > 0:
        shifted = shifted / max_dist
        
    if handedness == "Left":
        shifted[:, 0] = -shifted[:, 0]
        
    return shifted.flatten()

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
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

def main():
    st.title("ASL Alphabet Recognition – Both Hands")
    
    detector = load_detector()
    model, model_source, status_msg = load_asl_model()
    
    st.sidebar.title("Model Information")
    st.sidebar.write("This application uses a fully **hand-agnostic** neural network. It can recognize signs whether you use your Left or Right hand!")
    st.sidebar.write("---")
    
    if model_source == "smart_gestures":
        st.sidebar.success("✅ Powered by `smart_gestures` open-source pre-trained model!")
    elif model_source == "local_h5":
        st.sidebar.info("✅ Powered by locally trained `asl_hand_landmark_model.h5`!")
    else:
        st.sidebar.error("❌ Running on Dummy Model!")
        st.error(f"⚠️ **WARNING:** {status_msg}")
        st.info("To make real predictions, run `pip install smart_gestures` (if available) OR run `python train_model.py` to train your own real model.")
    
    st.sidebar.write("---")
    st.sidebar.write("Instructions:")
    st.sidebar.write("1. Check 'Start Webcam'.")
    st.sidebar.write("2. Make ASL alphabet gestures (A-Z).")
    
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.empty()
    
    if run:
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
                    
                    features = process_landmarks(hand_landmarks, handedness)
                    
                    if model_source == "smart_gestures":
                        # Convert to format expected by smart_gestures
                        # Assuming it expects the 42-element vector or can handle it directly.
                        try:
                            # Depending on package API, it might just return the letter directly
                            prediction = model.predict(features)
                            if isinstance(prediction, str):
                                predicted_letter = prediction
                            else:
                                class_idx = np.argmax(prediction)
                                predicted_letter = CLASSES[class_idx]
                        except Exception:
                            # Fallback if API differs slightly
                            predicted_letter = "?"
                    else:
                        # Local or dummy Keras model
                        prediction = model.predict(np.array([features]), verbose=0)
                        class_idx = np.argmax(prediction[0])
                        predicted_letter = CLASSES[class_idx]
                    
                    label_text = f"{handedness}: {predicted_letter}"
                    draw_landmarks_and_box(frame_rgb, hand_landmarks, label_text)
                    
            FRAME_WINDOW.image(frame_rgb)
            
        cap.release()
    else:
        st.write("Webcam is stopped.")

if __name__ == "__main__":
    main()
