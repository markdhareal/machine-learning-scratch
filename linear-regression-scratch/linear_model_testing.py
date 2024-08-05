from sklearn.model_selection import train_test_split
from linear_regression_model import LinearRegressionClass
from sklearn import datasets
import numpy as np

def mean_squared_error(y_test, predictions):
    return np.mean((y_test - predictions) ** 2)

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regression_model = LinearRegressionClass()

regression_model.fit(X_train, y_train)

predictions = regression_model.predict(X_test)

print(predictions)

print(mean_squared_error(y_test, predictions))