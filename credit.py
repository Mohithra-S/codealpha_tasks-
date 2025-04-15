import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset (replace with actual dataset path)
import pandas as pd

data = pd.read_csv(r"C:\\Users\\MOHITHRA\\OneDrive\\Desktop\\Ml one month  intern\\credit_data.csv")
print(data.head())  # To check if the file loads correctly

# Preprocess data
data.dropna(inplace=True)  # Handle missing values

# Encode categorical variables (check if these columns exist)
label_encoders = {}
categorical_columns = ['employment_status', 'marital_status']
for col in categorical_columns:
    if col in data.columns:  # Ensure column exists
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Separate features and target variable
target_column = 'creditworthy'  # Ensure this column exists
if target_column not in data.columns:
    raise ValueError(f"Column '{target_column}' not found in dataset")

X = data.drop(columns=[target_column])
y = data[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
