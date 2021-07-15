from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from src.regression.linear_regressors import LinearRegressor
from sklearn.linear_model import LinearRegression
from src.utility.util import plot_loss
from sklearn.metrics  import mean_squared_error


#load diabetes data
X, Y = load_diabetes(return_X_y=True)

#split data for train test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# instantiate and train using a dummy model
dummy_model = DummyRegressor(strategy='mean')
dummy_model.fit(X_train, Y_train)


# instantiate and train using a custom MLR model
model = LinearRegressor(iterations=5000, learning_rate=0.05)
model.fit(X_train, Y_train)



#instantiate and train using  sklearn
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, Y_train)



#plot loss over time for custom model
plot_loss(model)


# comparing Coefficients
print(f"Coefficients using custom model {model.weights}")
print(f"Coefficients using Sklearn model {model_sklearn.coef_}")

print(f"Biases using custom model {model.bias}")
print(f"Biases using Sklearn model {model_sklearn.intercept_}")



# Compare the MSE of models
print("Dummy model test data RMSE : ",mean_squared_error(Y_test,dummy_model.predict(X_test),squared=False))
print("Sklearn model test data RMSE : ",mean_squared_error(Y_test,model_sklearn.predict(X_test),squared=False))
print("Custom model test data RMSE : ",mean_squared_error(Y_test,model.predict(X_test),squared=False))



