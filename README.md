# Real-Time ASL Alphabet Sign Language Detection (Production-Ready)

A robust, production-grade American Sign Language (ASL) alphabet (A-Z) recognition system. It uses your webcam to detect and classify gestures from both **left and right hands** simultaneously.

## Key Technical Features

1. **3D Depth Integration (Z-Axis):** Extracts 63 features per hand (`x`, `y`, `z` coordinates for 21 landmarks) to provide true depth context, greatly improving accuracy on overlapping gestures (like R, U, V).
2. **TensorFlow Lite Engine:** Uses a `.tflite` model for blazing fast CPU inference, ensuring ultra-low latency real-time performance.
3. **Temporal Smoothing:** Implements a rolling window mode filter (collections.deque) over consecutive frames to completely eliminate prediction flickering.
4. **Hand-Agnostic Processing:** Horizontally mirrors left-hand X-coordinates during feature extraction, allowing a single neural network to process both hands interchangeably.
5. **Confidence Scoring:** Extracts Softmax output probabilities to show real-time percentage confidence for every prediction.

## Requirements

- Python 3.9+
- See `requirements.txt` for specific packages.

## Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate the Model:**
   Since this uses a custom 63-feature TFLite model, you must train it locally first. The script will automatically download the 87k ASL images from Kaggle and train the TFLite model.
   ```bash
   python train_model.py
   ```
3. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## Verification

Run the test suite to ensure all components (Normalization, Smoothing, TFLite inference) are functioning correctly:
```bash
python check_project.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
