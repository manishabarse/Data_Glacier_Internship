# Breast Cancer Classification Assignment

## Introduction
This assignment aims to develop a machine learning model for breast cancer classification using the Breast Cancer Wisconsin (Diagnostic) Dataset. The dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass, and the task is to classify the diagnosis as benign or malignant.

## Files
1. `preprocess_data.py`: Python script for data preprocessing, including handling missing values, scaling features, and splitting the dataset into training and testing sets.
2. `train_model.py`: Python script for training the machine learning model on the preprocessed data.
3. `app.py`: Flask application script for deploying the trained model as a web application. Includes an API endpoint for making predictions.
4. `send_request.py`: Python script for sending POST requests to the Flask API endpoint to make predictions.

## Data Preprocessing
The `preprocess_data.py` script performs the following steps:
- Data loading and preprocessing
- Handling missing values
- Scaling features
- Splitting the dataset into training and testing sets

## Model Training
The `train_model.py` script trains a Support Vector Machine (SVM) classifier on the preprocessed data. It includes:
- Model training
- Model evaluation on the test set
- Reporting accuracy, precision, recall, and F1-score

## Flask Deployment
The `app.py` script deploys the trained SVM model as a Flask web application. It includes:
- Creation of the Flask application
- API endpoint (`/predict`) for making predictions using the trained model

## Sending Requests
The `send_request.py` script sends POST requests to the Flask API endpoint (`/predict`) to make predictions on new data.

## Usage
To run the project:
1. Execute `preprocess_data.py` to preprocess the data.
2. Run `train_model.py` to train the model and evaluate its performance.
3. Start the Flask application by running `app.py`.
4. Use `send_request.py` to send POST requests and make predictions.

## Dependencies
- Python 3.x
- Libraries: scikit-learn, Flask, pandas, numpy, seaborn, matplotlib
