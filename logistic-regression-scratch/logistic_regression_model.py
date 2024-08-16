import numpy as np

def sigmoid(linear):
    return 1 / (1 + np.exp(-linear))
class LogisticRegression():

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):

            linear_prediction = np.dot(X, self.weights) + self.bias
            prediction = sigmoid(linear_prediction)

            dw = (1/n_samples) * np.dot(X.T, (prediction - y))
            db = (1/n_samples) * np.sum(prediction - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        linear_prediction = np.dot(X, self.weights) + self.bias
        prediction = sigmoid(linear_prediction)
        final_pred = [0 if y < 0.5 else 1 for y in prediction]
        return final_pred