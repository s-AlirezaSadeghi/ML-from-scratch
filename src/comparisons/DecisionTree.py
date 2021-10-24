from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier as sk_decision_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from src.utility.metrics import accuracy
from sklearn.dummy import DummyClassifier
from src.cart.decision_tree import ClassificationTree
from sklearn.datasets import load_breast_cancer

# Generate data
data = load_breast_cancer()
data_full_label = data["feature_names"].tolist() + data["target_names"].tolist()

X, y = data["data"], data['data']


# split data for train test
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# instantiate and train using a dummy model
dummy_model = DummyClassifier(strategy='prior')
dummy_model.fit(X_train, Y_train)

# instantiate and train using a ml-from-scratch MLR model
decision_tree = ClassificationTree(data_full_label)
decision_tree.fit(X_train, Y_train)


# instantiate and train using  sklearn
model_sk = sk_decision_tree()
model_sk.fit(X_train, Y_train)

decision_tree.print_tree(decision_tree.tree)
decision_tree.predict(X_test[0],decision_tree.tree)


custom_accuracy = accuracy(decision_tree.predict(X_test), Y_test)
sk_accuracy = accuracy(model_sk.predict(X_test), Y_test)
dummy_accuracy = accuracy(dummy_model.predict(X_test), Y_test)

print(f"Accuracy of  ml-from-scratch model {custom_accuracy}")
print(f"Accuracy of  dummy-classifier model {dummy_accuracy}")
print(f"Accuracy of  Sklearn model {sk_accuracy}")




tn, fp, fn, tp = confusion_matrix(decision_tree.predict(X_test), Y_test).ravel()
print(f'ml-from-scratch model True Negative: {tn} , False Positive: {fp} , False negative: {fn} , True Positive: {tp} ')

tn, fp, fn, tp = confusion_matrix(model_sk.predict(X_test), Y_test).ravel()
print(f'Sklearn model True Negative: {tn} , False Positive: {fp} , False negative: {fn} , True Positive: {tp} ')

tn, fp, fn, tp = confusion_matrix(dummy_model.predict(X_test), Y_test).ravel()
print(
    f'dummy-classifier model True Negative: {tn} , False Positive: {fp} , False negative: {fn} , True Positive: {tp} ')