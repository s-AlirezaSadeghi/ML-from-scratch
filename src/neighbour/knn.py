from collections import Counter
import math
from abc import ABC, abstractmethod
import numpy as np
from src.utility.metrics import euclidean_distance

class BaseKnn(ABC):
    def __init__(self, value):
        self.value = value
        super().__init__()

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray):
        pass

    @abstractmethod
    def predict(self):
        pass


class KnnClassifier(BaseKnn):
    def __init__(self, k=3):
        self.k = k
        self.distance_index = []
        self.y_train = None
        self.X_train = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.X_train = X
        self.y_train = Y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # compute distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # distances = [np.linalg.norm(x-x_train) for x_train in self.X_train]
        self.distances = distances
        # get k-nearest samples , labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority votes/ Mode
        majority_vote = Counter(k_nearest_labels).most_common(1)[0][0]
        return majority_vote


