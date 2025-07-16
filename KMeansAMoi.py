import numpy as np

class KMeans:
    def __init__(self, k, max_iters=100, tol=0.0001):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol  # tolérance pour la convergence
        self.centroids = None
        self.labels = None

    def fit(self, X):
        # Choisir aléatoirement k points comme centroïdes initiaux
        np.random.seed(31)
        random_indices = np.random.permutation(X.shape[0])[:self.k]
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # Étape 1 : assigner les points au centroïde le plus proche
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)

            # Étape 2 : recalculer les centroïdes
            new_centroids = np.array([
                X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else self.centroids[i]
                for i in range(self.k)
            ])

            # Vérifier la convergence
            shift = np.linalg.norm(self.centroids - new_centroids)
            if shift < self.tol:
                break

            self.centroids = new_centroids
            self.labels = labels

    def predict(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _compute_distances(self, X):
        # Calculer la distance euclidienne entre chaque point et les centroïdes
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)