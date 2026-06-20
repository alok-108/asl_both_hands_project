# Real-Time ASL Alphabet Sign Language Detection with Both Hands

A Streamlit application that detects and recognizes American Sign Language (ASL) alphabets in real-time. It uses MediaPipe to extract hand landmarks and prioritizes the open-source **`smart_gestures`** package to accurately classify the gestures.

## Features
- Real-time webcam processing using Streamlit.
- Support for **both Left and Right hands** independently. The model is hand-agnostic (using wrist-normalized and x-flipped coordinates for the left hand).
- Automatically attempts to load the true ASL model from the `smart_gestures` PyPI package, requiring zero downloading of datasets or training!
- If `smart_gestures` is unavailable, it gracefully falls back to a locally trained model (`asl_hand_landmark_model.h5`), and finally falls back to a dummy model to prevent crashes.

## Setup Instructions

1. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(This will attempt to install the `smart_gestures` package to provide the open-source model)*

2. Run the Web App:
   ```bash
   streamlit run app.py
   ```
   *Streamlit will open in your default browser. Click the "Start Webcam" checkbox to begin real-time recognition.*

## Custom Fallback Training
If `smart_gestures` is not working for you and you wish to train your own local neural network using the massive Kaggle ASL Alphabet dataset:
1. Run `python train_model.py`.
2. If you don't have the dataset, the script will automatically download the 87,000 images using `kagglehub`!
3. The script will cache the extracted landmarks into `X.npy` and `y.npy` to drastically speed up future runs, and it will train a feed-forward NN for at least 15 epochs.
4. The final, real model will be saved as `asl_hand_landmark_model.h5`, which `app.py` will automatically detect and use as its secondary fallback.
