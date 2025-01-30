from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample dataset
data = {'Tenure': [2, 10, 5, 1, 8], 
        'Monthly_Charges': [50, 100, 80, 30, 120], 
        'Churn': [1, 0, 0, 1, 0]}
df = pd.DataFrame(data)

X = df[['Tenure', 'Monthly_Charges']]
y = df['Churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Predict & Evaluate
y_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
