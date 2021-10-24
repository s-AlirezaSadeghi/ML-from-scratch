from matplotlib import pyplot as plt
from src.utility.metrics import accuracy
from sklearn.datasets import make_classification, make_moons
from src.regression.linear_regressors import LogisticRegression
from sklearn.linear_model import LogisticRegression as sk_logi
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier

# Generate data
X, y = make_classification(n_features=5, n_redundant=0,
                           n_informative=2, random_state=101,
                           n_clusters_per_class=1)

# split data for train test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# instantiate and train using a dummy model
dummy_model = DummyClassifier(strategy='prior')
dummy_model.fit(X_train, Y_train)

# instantiate and train using a ml-from-scratch MLR model
model = LogisticRegression(verbose=False, learning_rate=0.05, iterations=12000)
model.fit(X_train, Y_train)

# instantiate and train using  sklearn
model_sk = sk_logi()
model_sk.fit(X_train, Y_train)

# comparing Coefficients
print(f"Coefficients using ml-from-scratch model {model.weights}")
print(f"Coefficients using Sklearn model {model_sk.coef_[0]}")

print(f"Biases using ml-from-scratch model {model.bias}")
print(f"Biases using Sklearn model {model_sk.intercept_[0]}")

custom_accuracy = accuracy(model.predict(X_test), Y_test)
sk_accuracy = accuracy(model_sk.predict(X_test), Y_test)
dummy_accuracy = accuracy(dummy_model.predict(X_test), Y_test)

print(f"Accuracy of  dummy-classifier model {dummy_accuracy}")
print(f"Accuracy of  ml-from-scratch model {custom_accuracy}")
print(f"Accuracy of  Sklearn model {sk_accuracy}")

tn, fp, fn, tp = confusion_matrix(model.predict(X_test), Y_test).ravel()
print(f'ml-from-scratch model True Negative: {tn} , False Positive: {fp} , False negative: {fn} , True Positive: {tp} ')

tn, fp, fn, tp = confusion_matrix(model_sk.predict(X_test), Y_test).ravel()
print(f'Sklearn model True Negative: {tn} , False Positive: {fp} , False negative: {fn} , True Positive: {tp} ')

tn, fp, fn, tp = confusion_matrix(dummy_model.predict(X_test), Y_test).ravel()
print(
    f'dummy-classifier model True Negative: {tn} , False Positive: {fp} , False negative: {fn} , True Positive: {tp} ')
