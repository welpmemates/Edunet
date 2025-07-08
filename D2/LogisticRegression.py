import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import   classification_report, accuracy_score

# Step 1: Load the CSV (replace with your real filename)
df = pd.read_csv("solar_performance_all_seasons.csv")  # Update with actual path

# Step 2: Define features (X) and target (y)
X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle','kwh']]
y = df['season']

# Step 3: Encode the categorical target into numeric labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # e.g., summer -> 2, winter -> 1, etc.

# Step 4: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets to verify the split
print("X_train: " , X_train.shape)
print("X_test: " , X_test.shape)
print("y_train: " , y_train.shape)
print("y_test: " , y_test.shape)

# Step 5: Train Logistic Regression model
model = LogisticRegression(max_iter=10)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Save y_test, y_pred and le for plotting
np.save('y_test_2.npy', y_test)
np.save('y_pred_2.npy', y_pred)
np.save('class_labels.npy', le.classes_)

# Step 8: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))