import numpy as np
from numpy.typing import NDArray


class Solution:
    learning_rate = 0.01
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]) -> NDArray[np.float64]:
        
        N_initial_weights = len(initial_weights)
        X_length = len(X)
        for _ in range(num_iterations):
            prediction = self.get_model_prediction(X, initial_weights)
            for j in range(N_initial_weights):
                gradients = self.get_derivative(prediction, Y, X_length, X, j)
                initial_weights[j] -= gradients * self.learning_rate
        return np.round(initial_weights, 5)
        
