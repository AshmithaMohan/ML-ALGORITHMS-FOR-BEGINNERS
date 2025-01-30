from sklearn.cluster import KMeans
import numpy as np

# Sample dataset
data = np.array([[25, 30000], [40, 50000], [35, 45000], [50, 60000], [28, 32000]])

# Train K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data)

# Predict clusters
print("Cluster labels:", kmeans.labels_)
