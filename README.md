# Gesture-Recognition-System
Machine Learning based Gesture Recognition System for Control of Human Machine Interface

Gesture Recognition using Myo Armband

Project Overview:
This project implements a gesture recognition system using the Myo Armband. It leverages various machine learning and deep learning models to classify gestures based on electromyography (EMG) signals.

Features:
-Classification using SVM, Decision Trees, and Artificial Neural Networks (ANN)

-Deep Learning models using Keras

-Data visualization of model accuracy and loss

Directory Structure:

-Notebooks/: Jupyter notebooks for model training and evaluation

  Deep Learning.ipynb

  Gesture_SVM_DecisionTree_ANN.ipynb

-Scripts/: Python scripts for data processing and model deployment

  dl.py, gesture.py, main_loop.py, test1.py, test2.py

-Datasets/: Raw EMG data categorized by gesture type

-Models/: Pre-trained model (model.h5)

-Images/: Visualizations of model performance

Installation:

1.Clone the repository:

  git clone <repository-url>
  cd <repository-directory>

2.Install required dependencies:

  pip install -r requirements.txt

Usage:

-To train the model, run the respective Jupyter notebook or use the Python scripts:

  python main_loop.py

Data Collection:

The dataset is organized into various gestures, including:

  Closed Grip

  Index Finger

  Middle Finger

  Pinch

  Rest

Model Evaluation:

Model accuracy and loss are visualized in:

  Model-Accuracy.png

  Model-Loss.png

License:

This project is licensed under the MIT License.
