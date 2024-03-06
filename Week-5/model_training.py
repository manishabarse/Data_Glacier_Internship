import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import joblib

# Load the Breast Cancer dataset
breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = pd.Series(breast_cancer.target)

# Feature Engineering: Perform PCA to reduce dimensionality
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Model Selection and Optimization: Grid search with SVM and Random Forest
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}
param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}

svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)

# Data Augmentation: Bootstrap resampling
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
X_train_resampled, y_train_resampled = resample(X_train, y_train, replace=True, n_samples=len(X_train), random_state=42)

# Model Training
svm.fit(X_train_resampled, y_train_resampled)
rf.fit(X_train_resampled, y_train_resampled)

# Save the models
joblib.dump(svm, 'svm_model.joblib')
joblib.dump(rf, 'random_forest_model.joblib')

# Model Training Evaluation
y_pred_svm = svm.predict(X_test)
y_pred_rf = rf.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("Classification Report-SVM:")
print(classification_report(y_test, y_pred_svm))
print("Classification Report-RF:")
print(classification_report(y_test, y_pred_rf))
