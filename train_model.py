import os
import numpy as np
import cv2
import glob
import urllib.request
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import kagglehub

# Constants
MODEL_NAME = "asl_hand_landmark_model.h5"
X_NPY = "X.npy"
Y_NPY = "y.npy"

# ASL Alphabet 0-25 -> A-Z (Dataset includes 'del', 'nothing', 'space', we will filter for A-Z only)
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def get_hand_landmarker():
    if not os.path.exists('hand_landmarker.task'):
        print("Downloading hand_landmarker.task...")
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 
            'hand_landmarker.task'
        )
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1, 
        min_hand_detection_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)

def extract_features(img_path, detector):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)
    
    if not detection_result.hand_landmarks:
        return None
        
    landmarks = detection_result.hand_landmarks[0]
    handedness = detection_result.handedness[0][0].category_name # "Left" or "Right"
    
    # 1. Shift wrist to (0,0)
    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y
    
    shifted = []
    for lm in landmarks:
        shifted.append([lm.x - wrist_x, lm.y - wrist_y])
    shifted = np.array(shifted)
    
    # 2. Scale by max distance to any point
    distances = np.linalg.norm(shifted, axis=1)
    max_dist = np.max(distances)
    if max_dist > 0:
        shifted = shifted / max_dist
        
    # 3. If Left hand, flip x-coordinates to make it hand-agnostic
    if handedness == "Left":
        shifted[:, 0] = -shifted[:, 0]
        
    # 4. Flatten to 42-element vector
    return shifted.flatten()

def get_dataset_path():
    # If a local folder exists, use it
    if os.path.exists("asl_alphabet_train"):
        # The dataset inside kaggle might be doubly nested
        if os.path.exists(os.path.join("asl_alphabet_train", "asl_alphabet_train")):
            return os.path.join("asl_alphabet_train", "asl_alphabet_train")
        return "asl_alphabet_train"
        
    print("Local dataset not found. Downloading via kagglehub...")
    # Downloads to kaggle cache directory
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    
    dataset_dir = os.path.join(path, "asl_alphabet_train", "asl_alphabet_train")
    if os.path.exists(dataset_dir):
        return dataset_dir
    return os.path.join(path, "asl_alphabet_train")

def main():
    # 1. Load or Generate Dataset
    if os.path.exists(X_NPY) and os.path.exists(Y_NPY):
        print("Loading cached dataset from .npy files...")
        X = np.load(X_NPY)
        y = np.load(Y_NPY)
    else:
        dataset_dir = get_dataset_path()
        print(f"Extracting features from {dataset_dir}...")
        detector = get_hand_landmarker()
        X_list, y_list = [], []
        
        folders = os.listdir(dataset_dir)
        for folder in folders:
            if folder not in CLASSES:
                continue
            class_idx = CLASSES.index(folder)
            folder_path = os.path.join(dataset_dir, folder)
            img_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
            
            # Using subset of images if the dataset is massive, to speed up processing
            # (There are 3000 images per class. 500 is more than enough for landmarks)
            for img_path in img_paths[:500]: 
                feat = extract_features(img_path, detector)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(class_idx)
            print(f"Processed class {folder}")
            
        X = np.array(X_list)
        y = np.array(y_list)
        print(f"Saving extracted features to {X_NPY} and {Y_NPY}...")
        np.save(X_NPY, X)
        np.save(Y_NPY, y)
        
    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # 2. Train Model
    print("Building and training model...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(42,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(26, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train for at least 10 epochs
    model.fit(X, y, epochs=15, batch_size=32, validation_split=0.2)
    
    # Save Model
    model.save(MODEL_NAME)
    print(f"Real model trained and saved to {MODEL_NAME}")

if __name__ == "__main__":
    main()
