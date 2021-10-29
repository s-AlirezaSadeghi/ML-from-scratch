from sklearn.neighbors import KNeighborsClassifier
from src.neighbour.knn import KnnClassifier
from sklearn.dummy import DummyClassifier

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from src.utility.metrics import accuracy
import logging

logging.basicConfig(level=logging.INFO)

logging.info("Load breast cancer data from Sklearn")
iris = datasets.load_breast_cancer()

logging.info("Split Train/Test data")
X, y = iris["data"], iris["target"]
# split data for train test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=101)


logging.info("Fit a ml-from-scratch KnnClassifier with k=5 ")
knn = KnnClassifier(k=5)
knn.fit(X_train, Y_train)


logging.info("Train a dummy classifier as based line")
dummy_model = DummyClassifier(strategy='prior')
dummy_model.fit(X_train, Y_train)


logging.info("Train a Sklearn KnnClassifier with k=5 ")
sk = KNeighborsClassifier(n_neighbors=5)
sk.fit(X_train, Y_train)


logging.info("Predict Test data using ml-from-scratch model")
predictions = knn.predict(X_test)

logging.info("Predict Test data using dummy model")
dummy_predictions = dummy_model.predict(X_test)

logging.info("Predict Test data using Sklearn model")
sk_predictions = sk.predict(X_test)


logging.info("Calculate ml-from-scratch accuracy")
custom_accuracy = accuracy(predictions, Y_test)

logging.info("Calculate dummy accuracy")
dummy_accuracy = accuracy(dummy_predictions, Y_test)

logging.info("Calculate Sklearn accuracy")
sk_accuracy = accuracy(sk_predictions, Y_test)


logging.info(f"Accuracy of ml-from-scratch model {custom_accuracy}")
logging.info(f"Accuracy of dummy model {dummy_accuracy}")
logging.info(f"accuracy of sklearn model {sk_accuracy}")


logging.info("Computing confusion matrix for ml-from-scratch model")
tn, fp, fn, tp = confusion_matrix(predictions, Y_test).ravel()
logging.info(f'ml-from-scratch model True Negative: {tn} , False Positive: {fp} , False negative: {fn} , True Positive: {tp} ')

logging.info("Computing confusion matrix for dummy model")
tn, fp, fn, tp = confusion_matrix(dummy_predictions, Y_test).ravel()
logging.info(f'ml-from-scratch model True Negative: {tn} , False Positive: {fp} , False negative: {fn} , True Positive: {tp} ')

logging.info("Computing confusion matrix for Sklearn model")
tn, fp, fn, tp = confusion_matrix(sk_predictions, Y_test).ravel()
logging.info(f'Sklearn model True Negative: {tn} , False Positive: {fp} , False negative: {fn} , True Positive: {tp}')

