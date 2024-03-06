from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model
svm_model = joblib.load('svm_model.joblib')
rf_model = joblib.load('random_forest_model.joblib')

# Initialize Flask application
app = Flask(__name__)

# API endpoint for SVM model
@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    data = request.json  # Get data from request
    features = np.array(data['features']).reshape(1, -1)  # Extract features
    prediction = svm_model.predict(features)[0]  # Make prediction
    return jsonify({'prediction': int(prediction)})  # Return prediction as JSON

# API endpoint for Random Forest model
@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    data = request.json  # Get data from request
    features = np.array(data['features']).reshape(1, -1)  # Extract features
    prediction = rf_model.predict(features)[0]  # Make prediction
    return jsonify({'prediction': int(prediction)})  # Return prediction as JSON

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
