import numpy as np

# LINEAR REGRESSION CLASS
class LinearRegressionClass:

    def __init__(self, learning_rate = 0.001, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        y_predicted = np.dot(X, self.weights) + self.bias

        # GRADIENT DESCENT
        for _ in range(self.num_iterations):
            dw = (1/num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/num_samples) * np.sum(y_predicted - y)

            self.weights = (self.weights - self.learning_rate) * dw
            self.bias = (self.bias - self.learning_rate) * db



    def predict():
        pass