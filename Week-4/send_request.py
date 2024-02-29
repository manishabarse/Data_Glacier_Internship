import requests
import json
import pandas as pd

url = 'http://127.0.0.1:5000/predict'
headers = {'Content-Type': 'application/json'}

# Load preprocessed data
X_train_scaled = pd.read_csv('X_train.csv')

# Take the first row as sample data
sample_data = {'features': X_train_scaled.iloc[0].tolist()}

# Send POST request to Flask application
response = requests.post(url, data=json.dumps(sample_data), headers=headers)

# Process response
if response.status_code == 200:
    print(response.json())
else:
    print('Error:', response.status_code)