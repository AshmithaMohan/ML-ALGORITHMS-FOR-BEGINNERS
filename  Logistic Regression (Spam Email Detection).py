from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample dataset (Email words & spam labels)
data = {'Word_Count': [100, 120, 200, 150, 300, 250],
        'Contains_Link': [1, 0, 1, 0, 1, 0],
        'Spam': [1, 0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

X = df[['Word_Count', 'Contains_Link']]
y = df['Spam']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
