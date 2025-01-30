from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Sample dataset
data = {'Square_Feet': [1000, 1500, 800, 2000, 1800],
        'Bedrooms': [2, 3, 1, 4, 3],
        'Price': [200000, 300000, 150000, 400000, 350000]}
df = pd.DataFrame(data)

X = df[['Square_Feet', 'Bedrooms']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
