import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv('datasets/energy_consumption.csv', parse_dates=['Date'], index_col='Date')

# Train ARIMA model
model = ARIMA(data, order=(5,1,0))
model_fit = model.fit()

# Predictions
predictions = model_fit.forecast(steps=10)
print("Forecast:", predictions)