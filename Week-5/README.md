# Breast Cancer Classification and scRNA-seq Analysis

This project aims to develop a machine learning model for breast cancer classification using the Breast Cancer Wisconsin (Diagnostic) Dataset, perform scRNA-seq analysis, deploy a Flask web application for model inference, and set up the application on AWS.

## Model Training
1. **Model Selection:** SVM and Random Forest classifiers are trained on the Breast Cancer Wisconsin dataset.
2. **Feature Engineering:** Principal Component Analysis (PCA) is used to reduce dimensionality.
3. **Hyperparameter Optimization:** Grid search is performed to find the best parameters for the models.
4. **Model Evaluation:** Classification reports are generated to evaluate model performance.

## Single-cell RNA Sequencing (scRNA-seq) Analysis
1. **Data Preprocessing:** The scRNA-seq data is loaded and preprocessed using Scanpy.
2. **Quality Control:** Filtering, normalization, and identification of highly variable genes are performed.
3. **Dimensionality Reduction:** PCA is applied to reduce the dimensionality of the data.
4. **Clustering:** Cell clustering is performed using the Leiden algorithm.
5. **Differential Expression Analysis:** Marker genes for each cluster are identified.

## Flask Application Deployment
1. **App Setup:** Flask application is developed with routes for model prediction.
2. **Model Serialization:** Trained SVM and Random Forest models are serialized using joblib.
3. **Deployment:** The Flask application is deployed on an AWS EC2 instance.

## AWS Setup
1. **EC2 Instance:** An EC2 instance is launched on AWS to host the Flask application.
2. **Security Group:** Inbound rules are configured to allow traffic on port 5000 for Flask application access.
3. **File Upload:** Flask application files and serialized models are uploaded to the EC2 instance using SCP or SFTP.
4. **Application Deployment:** Flask application is started on the EC2 instance, and the public IP address is used to access the application.

## Repository Structure
- `model_training`: Contains scripts for model training and evaluation.
- `scRNA_analysis`: Includes Jupyter notebooks for scRNA-seq analysis.
- `flask_app`: Houses Flask application files (`app.py`) and serialized models (`svm_model.joblib`, `random_forest_model.joblib`).

## Dependencies
- scikit-learn
- Scanpy
- Flask
- NumPy
- Pandas
- Matplotlib
- Joblib


