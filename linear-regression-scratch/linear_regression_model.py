import numpy as np

# LINEAR REGRESSION CLASS
class LinearRegressionClass:

    def __init__(self, learning_rate = 0.001, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        y_predicted = (self.weights * X) + self.bias

        # GRADIENT DESCENT
        


    def predict():
        pass