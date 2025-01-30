import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
dates = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
prices = np.array([100, 105, 102, 108, 110, 115])

# Train Linear Regression
model = LinearRegression()
model.fit(dates, prices)

# Predict future prices
future_dates = np.array([7, 8, 9]).reshape(-1, 1)
future_prices = model.predict(future_dates)

# Plot
plt.scatter(dates, prices, color='blue', label="Actual Prices")
plt.plot(future_dates, future_prices, color='red', linestyle='dashed', label="Predicted Prices")
plt.legend()
plt.show()
