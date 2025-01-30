from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()
X = digits.data

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Plot reduced data
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=digits.target, cmap='jet', alpha=0.7)
plt.colorbar()
plt.show()
