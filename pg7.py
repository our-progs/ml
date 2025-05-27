import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing(as_frame=True)
X = housing.data[["AveRooms"]]
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

comparision = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", label="Predicted")
plt.xlabel("Average number of rooms (AveRooms)")
plt.ylabel("Median value of homes ($100,000)")
plt.title("Linear Regression - California Housing Dataset")
plt.legend()
plt.show()

print("Linear Regression - California Housing Dataset")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

file_path = "auto-mpg.csv"
df_auto_mpg = pd.read_csv(file_path)
df_auto_mpg["horsepower"] = pd.to_numeric(df_auto_mpg["horsepower"], errors='coerce')
df_auto_mpg.dropna(inplace=True)

X_poly = df_auto_mpg[['horsepower']].astype(float)
Y_poly = df_auto_mpg['mpg'].astype(float)

X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y_poly, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, Y_train)
Y_pred_poly = poly_reg.predict(X_test_poly)

mse_poly = mean_squared_error(Y_test, Y_pred_poly)
print("Polynomial Regression (Degree 2) MSE:", mse_poly)

sorted_indices = np.argsort(X_test.values.flatten())
X_test_sorted = X_test.values.flatten()[sorted_indices]
Y_test_sorted = Y_test.values.flatten()[sorted_indices]
Y_pred_sorted = Y_pred_poly[sorted_indices]

plt.figure(figsize=(8, 5))
plt.scatter(X_test, Y_test, color='blue', label="Actual MPG")
plt.plot(X_test_sorted, Y_pred_sorted, color='red', linewidth=2, label="Polynomial Regression (Degree 2)")
plt.xlabel("Horsepower")
plt.ylabel("MPG (Fuel Efficiency)")
plt.title("Polynomial Regression - Auto MPG Dataset")
plt.legend()
plt.show()
