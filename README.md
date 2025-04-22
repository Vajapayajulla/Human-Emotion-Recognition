
# Human Emotion Recognition Using Facial Features

## Project Overview
This project focuses on building a machine learning model to classify human emotions from facial expressions using a hybrid deep learning architecture that combines **LSTM (Long Short-Term Memory)** and **CNN (Convolutional Neural Network)**. The model is trained to detect emotions in real-time from images or webcam input.

Achieved an accuracy of **80.32%** during testing, demonstrating strong performance for real-time emotion detection tasks.

---

## Features
- 📸 Real-time emotion detection using webcam.
- 🧠 Hybrid LSTM-CNN deep learning model.
- 📈 Achieved 80.32% accuracy on validation datasets.
- 🔥 Preprocessing techniques for better feature extraction from facial landmarks.

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV (for image and video processing)
- NumPy, pandas
- Matplotlib (for visualization)

---

## Dataset
The model was trained on standard facial expression datasets such as:
- FER-2013 (Facial Expression Recognition 2013)
- [Or mention another dataset if used]

---

## Model Architecture
- **Convolutional Neural Networks (CNN)** are used to extract spatial features from facial images.
- **Long Short-Term Memory Networks (LSTM)** capture temporal dependencies between sequences of facial features.
- An **Attention Mechanism** was explored for better focus on relevant features (optional mention if applied).

---

## Installation
```bash
git clone https://github.com/yourusername/human-emotion-recognition.git
cd human-emotion-recognition
pip install -r requirements.txt
```

---

## How to Run
1. Train the model (if not using pretrained weights):
```bash
python train.py
```
2. For real-time emotion detection:
```bash
python detect_emotion.py
```

---

## Results
| Metric       | Value    |
| ------------ | -------- |
| Accuracy     | 80.32%   |
| Model Size   | ~20 MB   |
| Inference Time | ~40 ms per frame |

Sample detected emotions:
![Sample Emotions](path_to_sample_images)

---

## Future Work
- Improve accuracy by using deeper architectures like ResNet.
- Deploy the model on mobile devices using TensorFlow Lite.
- Extend to multi-emotion recognition (detect multiple emotions simultaneously).

---

## Contributors
- **Vajapayajulla Bhuvaneshwari** – Project Lead

---

# 🚀 Let's make AI understand human emotions better!
