## Overview
Drowsiness Detection system that can detect whether a person is sleeping or awake in real-time through webcam face detection, with an alarm to wake up the drowsy person! 

## Description
This Computer Vision model implements a sophisticated image processing and prediction pipeline using Convolutional Neural Networks (CNNs) in TensorFlow and Keras. The primary goal is to achieve high-accuracy image classification through meticulous data preprocessing, model architecture design, and performance evaluation. Key components include:

Data Preprocessing: Utilizes OpenCV for image manipulation, including resizing, normalization, and augmentation (e.g., rotation, flipping, zooming) to enhance model generalization and robustness against overfitting.

CNN Model Architecture: Constructs a deep learning model with multiple convolutional layers, batch normalization, dropout for regularization, and fully connected layers. The model leverages ReLU activations and softmax for multi-class classification.

Model Training and Optimization: Employs advanced techniques such as early stopping and learning rate scheduling. The Adam optimizer is used to minimize categorical cross-entropy loss, ensuring efficient and effective convergence.

Evaluation and Metrics: Implements comprehensive model evaluation using accuracy, precision, recall, and F1-score metrics. The model's performance is validated on a separate test dataset, with confusion matrices and ROC curves to analyze classification results.

Prediction and Deployment: Integrates a streamlined pipeline for preprocessing new images and generating predictions. This includes loading saved model weights, applying the same preprocessing steps, and outputting class probabilities with corresponding confidence scores.

Visualization and Interpretation: Uses Matplotlib for visualizing training history, including loss and accuracy plots. Additionally, Grad-CAM or similar techniques may be applied to interpret model decisions, providing insights into which parts of the image are influential for predictions.

## Technologies Used
Programming Language: Python

Data Processing and Analysis:
    NumPy
    Pandas
    OpenCV
    
Machine Learning and Deep Learning:
    TensorFlow
    Keras
    
Visualization:
    Matplotlib
    
Jupyter Notebook: Used for interactive code development and visualization.

## Project Structure

Data Preprocessing:
    Loading and inspecting the dataset.
    Resizing and normalizing images.
    Data augmentation techniques.

Model Development:
    Building a Convolutional Neural Network (CNN) using Keras.
    Compiling the model with appropriate loss functions and optimizers.
    Training the model on the dataset.

Model Evaluation:
    Evaluating the model performance on the validation set.
    Generating and plotting metrics such as accuracy and loss.

Prediction:
    Loading and preprocessing new images.
    Using the trained model to make predictions on new data.


## How to Use

1. Clone the repository:
git clone https://github.com/yourusername/your-repository.git    
cd your-repository

3. Install the required dependencies:
pip install -r requirements.txt

4. Run the Jupyter Notebook:
jupyter notebook Real-time-detection.ipynb

Note: Make sure to turn on the webcam!!

## References
TensorFlow Documentation: TensorFlow
Keras Documentation: Keras
OpenCV Documentation: OpenCV
Matplotlib Documentation: Matplotlib
