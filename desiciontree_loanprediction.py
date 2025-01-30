import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample dataset (Loan Approval)
data = {'Income': [3000, 4500, 6000, 8000, 12000],
        'Credit_Score': [600, 650, 700, 750, 800],
        'Approved': [0, 0, 1, 1, 1]}
df = pd.DataFrame(data)

X = df[['Income', 'Credit_Score']]
y = df['Approved']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict & Evaluate
y_pred = clf.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
