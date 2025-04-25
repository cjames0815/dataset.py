# Decision Tree Classification Starter Code

# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score  # fixed line break error

# Step 2: Import the Dataset
# Replace 'your_dataset.csv' with the actual file name or path
df = pd.read_csv('your_dataset.csv')

# Preview the dataset
print("Dataset Preview:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Step 3: Select Feature and Target Variables
# Replace 'feature1', 'feature2', etc. and 'target_column' with actual column names
features = df[['feature1', 'feature2', 'feature3']]  # update with real feature names
target = df['target_column']  # update with the correct target column

# Step 4: Split the Dataset (70% Training, 30% Testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Step 5: Initialize and Fit a Decision Tree Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Step 7: Compute the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Step 8: Compute and print accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)

# Use 'binary' only if it's a binary classification; otherwise, use 'macro' or 'weighted'
precision = precision_score(y_test, y_pred, average='binary')  
recall = recall_score(y_test, y_pred, average='binary')

print("\nPerformance Metrics:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
