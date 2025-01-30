from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
emails = ["Congratulations, you've won a free iPhone!", 
          "Reminder: Your bank statement is ready.", 
          "Earn money fast with this simple trick.", 
          "Meeting scheduled at 3 PM."]
labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Convert text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train Naïve Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Predict & Evaluate
y_pred = nb.predict(X_test)
print("Naïve Bayes Accuracy:", accuracy_score(y_test, y_pred))
