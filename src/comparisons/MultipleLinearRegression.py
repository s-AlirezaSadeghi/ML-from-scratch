from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from src.regression.linear_regressors import LinearRegressor
from sklearn.linear_model import LinearRegression
from src.utility.util import plot_loss
from sklearn.metrics import mean_squared_error
import logging

# plot loss graph
GRAPH = False

logging.basicConfig(level=logging.INFO)

# load diabetes data
X, Y = load_diabetes(return_X_y=True)

# split data for train test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# instantiate and train using a dummy model
dummy_model = DummyRegressor(strategy='mean')
dummy_model.fit(X_train, Y_train)

# instantiate and train using a ml-from-scratch MLR model
model = LinearRegressor(iterations=5000, learning_rate=0.05)
model.fit(X_train, Y_train)

# instantiate and train using  sklearn
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, Y_train)

# plot loss over time for ml-from-scratch model
if GRAPH:
    plot_loss(model)

# comparing Coefficients
logging.info(f"Coefficients using ml-from-scratch model {model.weights}")
logging.info(f"Coefficients using Sklearn model {model_sklearn.coef_}")

logging.info(f"Biases using ml-from-scratch model {model.bias}")
logging.info(f"Biases using Sklearn model {model_sklearn.intercept_}")

# Compare the MSE of models
logging.info(f"ml-from-scratch model test data RMSE : {mean_squared_error(Y_test, model.predict(X_test), squared=False)}")
logging.info(f"Dummy model test data RMSE : {mean_squared_error(Y_test, dummy_model.predict(X_test), squared=False)}")
logging.info(f"Sklearn model test data RMSE : {mean_squared_error(Y_test, model_sklearn.predict(X_test), squared=False)}")




