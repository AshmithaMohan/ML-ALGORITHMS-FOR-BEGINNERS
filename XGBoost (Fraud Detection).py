import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample dataset
data = {'Transaction_Amount': [500, 2000, 300, 5000, 100],
        'Account_Age': [5, 1, 8, 0.5, 10],
        'Fraudulent': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

X = df[['Transaction_Amount', 'Account_Age']]
y = df['Fraudulent']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
