import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        self.labels = np.argmin(distances, axis=1)

    def update_centroids(self, X):
        self.centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            old_centroids = np.copy(self.centroids)

            self.assign_clusters(X)
            self.update_centroids(X)

            if np.all(old_centroids == self.centroids):
                break

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)