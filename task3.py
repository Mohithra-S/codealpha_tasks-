# -*- coding: utf-8 -*-
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path to dataset
csv_path = r"C:\Users\MOHITHRA\OneDrive\Desktop\Ml one month  intern\CodeAlpha_Disease_Prediction_from_Medical_Data-main\Disease_symptom_and_patient_profile_dataset.csv"

# Create CSV if it doesn't exist
if not os.path.exists(csv_path):
    sample_data = {
        'Disease': ['Influenza', 'Common Cold', 'COVID-19', 'Influenza', 'Common Cold'],
        'Fever': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'Cough': ['Yes', 'Yes', 'Yes', 'No', 'Yes'],
        'Fatigue': ['Yes', 'Yes', 'Yes', 'Yes', 'No'],
        'Difficulty Breathing': ['Yes', 'No', 'Yes', 'No', 'No'],
        'Age': [30, 25, 40, 22, 28],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Female'],
        'Blood Pressure': ['High', 'Normal', 'High', 'Low', 'Normal'],
        'Cholesterol Level': ['High', 'Normal', 'High', 'Normal', 'Normal'],
        'Outcome Variable': ['Positive', 'Negative', 'Positive', 'Negative', 'Negative']
    }
    pd.DataFrame(sample_data).to_csv(csv_path, index=False)
    print(f"Sample dataset created at: {csv_path}")

# Load the dataset
df = pd.read_csv(csv_path)
print("\nLoaded Data:\n", df.head())

# Encode categorical variables
label_encoders = {}
for column in ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Features and Target
X = df.drop('Outcome Variable', axis=1)
y = df['Outcome Variable']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0)}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0)}")
print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0)}")

# Predict on new data
def predict_new(data_dict):
    new_data_encoded = {}
    for col, val in data_dict.items():
        if col in label_encoders:
            new_data_encoded[col] = label_encoders[col].transform([val])[0]
        else:
            new_data_encoded[col] = val
    new_data_df = pd.DataFrame([new_data_encoded], columns=X.columns)
    prediction = model.predict(new_data_df)
    result = label_encoders['Outcome Variable'].inverse_transform(prediction)
    return result[0]

# Test prediction 1
new_data_1 = {
    'Disease': 'Influenza',
    'Fever': 'Yes',
    'Cough': 'No',
    'Fatigue': 'Yes',
    'Difficulty Breathing': 'Yes',
    'Age': 20,
    'Gender': 'Female',
    'Blood Pressure': 'Low',
    'Cholesterol Level': 'Normal'
}
print("\nPrediction 1:", predict_new(new_data_1))

# Test prediction 2
new_data_2 = {
    'Disease': 'Common Cold',
    'Fever': 'No',
    'Cough': 'Yes',
    'Fatigue': 'Yes',
    'Difficulty Breathing': 'No',
    'Age': 25,
    'Gender': 'Female',
    'Blood Pressure': 'Normal',
    'Cholesterol Level': 'Normal'
}
print("Prediction 2:", predict_new(new_data_2))
