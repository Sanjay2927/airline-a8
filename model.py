# model_training.py

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("/Users/roshan/Desktop/SANJAY A8/Aircrash dataset.csv")

# Preview columns
print("Initial columns:", df.columns)

# ==== Step 1: Basic Preprocessing ====

# Example: Fill missing values (customize this part based on your dataset)
df.fillna(0, inplace=True)

# ==== Step 2: Define Target Variable ====

# Let's assume we predict whether fatalities occurred or not
df['Fatal_Flag'] = df['Fatalities'].apply(lambda x: 1 if x > 0 else 0)

# Drop irrelevant or text-heavy columns
X = df.drop(columns=['Fatalities', 'Fatal_Flag', 'Summary', 'Location', 'Operator', 'Flight #', 'Route', 'Registration'])
y = df['Fatal_Flag']

# Convert categorical columns to dummy variables
X = pd.get_dummies(X)

# ==== Step 3: Train/Test Split ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Step 4: Train the Model ====
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==== Step 5: Evaluate ====
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ==== Step 6: Save the model ====
with open("aircrash_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as aircrash_model.pkl")
