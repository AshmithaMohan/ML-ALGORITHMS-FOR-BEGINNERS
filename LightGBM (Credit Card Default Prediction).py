import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample dataset
data = {'Balance': [1000, 5000, 200, 7000, 1500],
        'Salary': [40000, 100000, 20000, 120000, 50000],
        'Default': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

X = df[['Balance', 'Salary']]
y = df['Default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("LightGBM Accuracy:", accuracy_score(y_test, y_pred))
