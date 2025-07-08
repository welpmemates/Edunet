import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load your CSV file
df = pd.read_csv("solar_performance_all_seasons.csv")  # Replace with your actual filename

# Step 2: Define features (X) and target (y)
X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']]
y = df['kwh']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets to verify the split
print("X_train: " , X_train.shape)
print("X_test: " , X_test.shape)
print("y_train: " , y_train.shape)
print("y_test: " , y_test.shape)

# Display the entire DataFrame to inspect its contents
# With df you can print different stats or contents just like mentioned in the gen.py
#print(df)

# Step 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 7: Save y_test and y_pred for plotting
np.save('y_test.npy', y_test)
np.save('y_pred.npy', y_pred)

# Step 8: Print model details and evaluation
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)