import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from logistic_regression_model import LogisticRegression

info = datasets.load_breast_cancer()
X, y = info.data, info.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

classifier = LogisticRegression(learning_rate=0.01)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print("Predictions: ")
print(y_pred)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)

acc = accuracy(y_pred, y_test)
print('Accuracy: ',acc)