import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (House Size vs. Price)
X = np.array([500, 700, 1000, 1200, 1500]).reshape(-1, 1)
Y = np.array([150000, 200000, 250000, 275000, 350000])

# Model training
model = LinearRegression()
model.fit(X, Y)

# Prediction
X_test = np.array([800, 1300]).reshape(-1, 1)
predictions = model.predict(X_test)

# Plot
plt.scatter(X, Y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', label="Regression Line")
plt.scatter(X_test, predictions, color='green', label="Predictions")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price ($)")
plt.legend()
plt.show()
