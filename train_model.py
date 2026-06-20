import os
import cv2
import numpy as np
import mediapipe as mp
import kagglehub
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Configuration
DATASET_DIR = "asl_alphabet_train/asl_alphabet_train"
MODEL_PATH_H5 = "asl_hand_landmark_model.h5"
MODEL_PATH_TFLITE = "asl_hand_landmark_model.tflite"
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") # 26 classes

def download_dataset():
    """Downloads dataset via kagglehub if not present."""
    if not os.path.exists(DATASET_DIR):
        print("Downloading Kaggle ASL Alphabet Dataset (87,000 images)... This may take a while.")
        path = kagglehub.dataset_download("grassknoted/asl-alphabet")
        global DATASET_DIR
        DATASET_DIR = os.path.join(path, "asl_alphabet_train/asl_alphabet_train")
        print(f"Dataset downloaded to {DATASET_DIR}")
    else:
        print("Dataset already found locally.")

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

def extract_features():
    """Extracts MediaPipe landmarks from dataset images and caches them."""
    if os.path.exists("X.npy") and os.path.exists("y.npy"):
        print("Loading cached features from X.npy and y.npy...")
        X = np.load("X.npy")
        y = np.load("y.npy")
        return X, y

    print("Extracting features using MediaPipe... (This will take a long time on 87,000 images)")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    X_list = []
    y_list = []
    
    for label_idx, letter in enumerate(CLASSES):
        folder_path = os.path.join(DATASET_DIR, letter)
        if not os.path.exists(folder_path):
            print(f"Warning: Directory {folder_path} not found. Skipping.")
            continue
            
        print(f"Processing class {letter}...")
        for filename in os.listdir(folder_path):
            if not filename.endswith(".jpg"):
                continue
                
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label # Left/Right
                
                features = process_landmarks(hand_landmarks, handedness)
                X_list.append(features)
                y_list.append(label_idx)
                
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    np.save("X.npy", X)
    np.save("y.npy", y)
    print(f"Extracted features for {len(X)} images and cached them.")
    return X, y

def train_and_convert_model(X, y):
    """Trains a feed-forward NN and converts it to TFLite."""
    print("Building and training the model...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(63,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(26, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Shuffle data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    model.fit(X, y, epochs=15, batch_size=64, validation_split=0.2)
    
    # Save legacy .h5 model just in case
    model.save(MODEL_PATH_H5)
    print(f"Legacy model saved to {MODEL_PATH_H5}")
    
    # Convert to TFLite
    print("Converting model to TFLite for production...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(MODEL_PATH_TFLITE, 'wb') as f:
        f.write(tflite_model)
    print(f"Production TFLite model saved to {MODEL_PATH_TFLITE}!")

if __name__ == "__main__":
    download_dataset()
    X, y = extract_features()
    if len(X) == 0:
        print("No features extracted. Exiting.")
    else:
        train_and_convert_model(X, y)
