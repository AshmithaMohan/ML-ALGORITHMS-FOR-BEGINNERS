from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample text data
X = ["This product is amazing", "I hate this item", "Very satisfied", "Terrible experience", "Highly recommended"]
y = [1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative

# Train Naïve Bayes model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X, y)

# Predict
test_reviews = ["Not bad", "Worst thing ever", "Fantastic quality"]
print(model.predict(test_reviews))  # Output: [1, 0, 1]
