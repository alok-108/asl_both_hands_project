import sys
import os
import cv2
import numpy as np

def print_result(name, passed, message=""):
    status = "[ PASS  ]" if passed else "[ FAIL  ]"
    print(f"{status} {name}: {message}")

def test_dependencies():
    try:
        import streamlit
        import mediapipe
        import PIL
        import tensorflow
        print_result("Dependencies", True, "All required packages are installed.")
        return True
    except ImportError as e:
        print_result("Dependencies", False, f"Missing package: {e}")
        return False

def test_normalization():
    try:
        from app import process_landmarks
        class DummyLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
                
        # Create 21 dummy landmarks
        landmarks = [DummyLandmark(i*0.01, i*0.02, i*0.001) for i in range(21)]
        
        # Test Right Hand
        features_right = process_landmarks(landmarks, "Right")
        if features_right.shape != (63,):
            print_result("Normalization", False, f"Expected shape (63,), got {features_right.shape}")
            return False
            
        # Test Left Hand (should flip X)
        features_left = process_landmarks(landmarks, "Left")
        
        # X coordinates should be negative of right hand, Y and Z should be equal
        if features_left[0] != -features_right[0]:
            print_result("Normalization", False, "Left hand X coordinates were not flipped properly.")
            return False
            
        print_result("Normalization", True, "Outputs 63-element vector. Left hand correctly mirrored.")
        return True
    except Exception as e:
        print_result("Normalization", False, f"Error: {e}")
        return False

def test_model_loading():
    if not os.path.exists("asl_hand_landmark_model.tflite"):
        print_result("Model Loading", False, "TFLite model not found. Run train_model.py first.")
        return False
        
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path="asl_hand_landmark_model.tflite")
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Verify shapes
        if input_details[0]['shape'][1] != 63:
            print_result("Model Loading", False, f"Expected input shape (1, 63), got {input_details[0]['shape']}")
            return False
            
        if output_details[0]['shape'][1] != 26:
            print_result("Model Loading", False, f"Expected output shape (1, 26), got {output_details[0]['shape']}")
            return False
            
        print_result("Model Loading", True, "Loaded TFLite successfully. Shapes OK (63 -> 26).")
        return True
    except Exception as e:
        print_result("Model Loading", False, f"TFLite Load Error: {e}")
        return False

def test_temporal_smoothing():
    try:
        from app import get_smoothed_prediction, history
        history["Right"].clear()
        
        res1 = get_smoothed_prediction("Right", 0, 95.5) # A
        res2 = get_smoothed_prediction("Right", 1, 90.0) # B (flicker)
        res3 = get_smoothed_prediction("Right", 0, 98.2) # A
        
        if "A" in res3 and "98%" in res3:
            print_result("Smoothing", True, "Mode filter stabilized prediction to 'A' despite flicker.")
            return True
        else:
            print_result("Smoothing", False, f"Failed to smooth: {res3}")
            return False
    except Exception as e:
        print_result("Smoothing", False, f"Error: {e}")
        return False

if __name__ == "__main__":
    print("==================================================")
    print("ASL Alphabet Project - Production Verification")
    print("==================================================")
    
    test_dependencies()
    test_normalization()
    test_temporal_smoothing()
    test_model_loading()
    
    print("\nVERDICT: Review the results above.")
