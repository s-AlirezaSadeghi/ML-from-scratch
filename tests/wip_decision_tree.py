import numpy as np
from src.classification.decision_tree_classifier import ClassificationTree
from src.classification.decision_tree_classifier import gini2
from src.classification.decision_tree_classifier import entropy_numpy
from src.classification.decision_tree_classifier import entropy_scipy
from collections import Counter

from src.utility import metrics as mt

# sample data creation
num_rows = 50
cardiac_data = np.random.randint(50, 112, size=(num_rows, 1), dtype=int)
age_data = np.random.randint(19, 85, size=(num_rows, 1), dtype=int)
heart_attack = np.random.choice([0, 1], size=(num_rows, 1), p=[0.75, 0.25])
training_data = np.column_stack((cardiac_data, age_data, heart_attack))


x = np.column_stack((cardiac_data, age_data))
y=  heart_attack

# calculate starting current_uncertainty
current_uncertainty = gini2(training_data[:, -1])
# print(f'initial information gain  is : {current_uncertainty}')
print(Counter(training_data[:, -1]))

# gini(training_data[:,-1])
mt.gini(training_data[:, -1])

# todo : these are different
entropy_scipy(training_data[:, -1])
entropy_numpy(training_data[:, -1])
mt.entropy(training_data[:, -1].tolist())


# ct = ClassificationTree(headers=['heart_rate','age','heart_attack'])
# ct.best_split(training_data)

ct = ClassificationTree(['heart_rate','age','heart_attack'])
ct.fit(x,y)
print('x')
# my_tree = tree_builder(data=training_data, labels=['heart_rate', 'age', 'heart_attack'])
print_tree(my_tree)
print(f'initial information gain  is : {current_uncertainty}')

print('Job done!')




print(training_data[0, :-1])
classify(training_data[0], my_tree)