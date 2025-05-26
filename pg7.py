# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------
# LINEAR REGRESSION ON CALIFORNIA HOUSING DATASET
# ------------------------------------------

# Load California housing dataset as a pandas DataFrame
housing = fetch_california_housing(as_frame=True)

# Extract 'AveRooms' as feature and 'target' (median house value) as label
X = housing.data[["AveRooms"]]
y = housing.target

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict target values for the test data
y_pred = model.predict(X_test)

# Compare predicted values with actual test target values
comparision = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

# Plotting the results: scatter for actual, line for predicted
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", label="Predicted")
plt.xlabel("Average number of rooms (AveRooms)")
plt.ylabel("Median value of homes ($100,000)")
plt.title("Linear Regression - California Housing Dataset")
plt.legend()
plt.show()

# Print performance metrics for the linear regression model
print("Linear Regression - California Housing Dataset")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Define the path to the dataset CSV file
file_path = "auto-mpg.csv"

# Load the Auto MPG dataset
df_auto_mpg = pd.read_csv(file_path)

# Convert 'horsepower' column to numeric, setting errors to NaN, then remove rows with missing values
df_auto_mpg["horsepower"] = pd.to_numeric(df_auto_mpg["horsepower"], errors='coerce')
df_auto_mpg.dropna(inplace=True)

# Extract the feature (horsepower) and target (mpg) as float
X_poly = df_auto_mpg[['horsepower']].astype(float)
Y_poly = df_auto_mpg['mpg'].astype(float)

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y_poly, test_size=0.2, random_state=42)

# Create polynomial features (degree 2 transformation)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Instantiate and train a Linear Regression model on polynomial-transformed data
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, Y_train)

# Predict target values using the polynomial regression model
Y_pred_poly = poly_reg.predict(X_test_poly)

# Compute and display Mean Squared Error
mse_poly = mean_squared_error(Y_test, Y_pred_poly)
print("Polynomial Regression (Degree 2) MSE:", mse_poly)

# Sort the test data for plotting a smooth curve
sorted_indices = np.argsort(X_test.values.flatten())
X_test_sorted = X_test.values.flatten()[sorted_indices]
Y_test_sorted = Y_test.values.flatten()[sorted_indices]
Y_pred_sorted = Y_pred_poly[sorted_indices]

# Plotting the results: scatter for actual, smooth curve for predicted
plt.figure(figsize=(8, 5))
plt.scatter(X_test, Y_test, color='blue', label="Actual MPG")
plt.plot(X_test_sorted, Y_pred_sorted, color='red', linewidth=2, label="Polynomial Regression (Degree 2)")
plt.xlabel("Horsepower")
plt.ylabel("MPG (Fuel Efficiency)")
plt.title("Polynomial Regression - Auto MPG Dataset")
plt.legend()
plt.show()
