import pandas as pd
from sklearn.svm import SVC
import joblib

# Load preprocessed data without headers
X_train = pd.read_csv('X_train.csv', header=None)
y_train = pd.read_csv('y_train.csv', header=None)[0]

# Train the model
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'breast_cancer_model.joblib')

print("Model training completed.")