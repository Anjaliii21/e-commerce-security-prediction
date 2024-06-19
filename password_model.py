import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Load the data (handle potential errors with 'errors=' argument)
try:
  data = pd.read_csv("pass.csv", on_bad_lines='skip')
except FileNotFoundError:
  print("Error: File 'pass.csv' not found.")
  exit()

# Display the first few rows of the dataframe (uncomment if needed)
# print(data.head())

df = pd.DataFrame(data)
df.info()

columns_to_remove = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6']

# Remove the specified columns
df = df.drop(columns=columns_to_remove, errors='ignore')

print("\nData After Removing Columns:")
print(df.head())

# Check if "strength" column exists before accessing it
if 'strength' in df.columns:
  data["strength"].unique()

# Remove null values (once is sufficient)
data = data.dropna().reset_index(drop=True)

# **Option 2: Vectorization** (using CountVectorizer for text data)
vectorizer = CountVectorizer(max_features=9)  # Adjust max_features as needed

X = vectorizer.fit_transform(data['password'])

# Label encoding for strength values
le = LabelEncoder()
y = le.fit_transform(data['strength'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {model.score(X_test, y_test)}")

cm = confusion_matrix(y_test, model.predict(X_test))
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
