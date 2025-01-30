from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Sample dataset
data = {'Glucose_Level': [80, 150, 100, 200, 90],
        'BMI': [25, 30, 22, 35, 27],
        'Diabetic': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

X = df[['Glucose_Level', 'BMI']]
y = df['Diabetic']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict & Evaluate
y_pred = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))

