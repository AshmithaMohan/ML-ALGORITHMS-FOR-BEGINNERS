
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample dataset
data = {'Credit_Score': [750, 600, 800, 500, 650],
        'Income': [70000, 40000, 80000, 30000, 50000],
        'Loan_Approved': [1, 0, 1, 0, 0]}
df = pd.DataFrame(data)

X = df[['Credit_Score', 'Income']]
y = df['Loan_Approved']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Predict & Evaluate the decision tree

y_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
