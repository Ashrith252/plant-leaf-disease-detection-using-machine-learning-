# Hybrid Ensemble Plant Leaf Disease Detection

## Overview
This project detects plant leaf diseases using a hybrid ensemble approach combining deep learning and classical machine learning models.

## Problem Statement
Single CNN-based models may fail on real-world images due to lighting variations and noise. This project improves reliability using an ensemble strategy.

## Approach
- CNN trained on RGB images
- CNN trained on enhanced (CLAHE) images
- Classical ML models (Random Forest, SVM, KNN)
- Final prediction using ensemble voting

## Tech Stack
- Python
- TensorFlow, Keras
- OpenCV
- Scikit-learn
- NumPy, Matplotlib

## Project Structure
- src/ → source code files
- models/ → trained model files
- outputs/ → output images and results
## How to Run
pip install -r requirements.txt  
python src/main.py
