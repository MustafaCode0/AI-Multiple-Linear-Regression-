import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
import numpy as np

# reading the data into df
data = pd.read_csv("Auto.csv")
df = pd.DataFrame(data)

# setting up multiple variables to x and y is mpg excluding mpg, name and origin.
x = df[["cylinders", "displacement", "horsepower", "weight", "acceleration", "year"]]
y = df["mpg"]

# splitting the data into 80/20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# declearing alpha values that i will be using and
# making r2 and lasso list to hold test results
alphas = [0, 1, 5, 10, 20, 30, 50, 100, 1000]
r2values = []
lassovalues = []

# making loop for ridge, training testing and adding score to list
for alp in alphas:
    rr = Ridge(alpha=alp)
    rr.fit(x_train, y_train)
    r2_test = metrics.r2_score(y_test, rr.predict(x_test))
    r2values.append(r2_test)

# making loop for lasso, training testing and adding score to list
for i in alphas:
    lso = Lasso(alpha=i)
    lso.fit(x_train, y_train)
    r2_lso = metrics.r2_score(y_test, lso.predict(x_test))
    lassovalues.append(r2_lso)

# plotting results and comparing them
plt.plot(alphas, r2values, label="Ridge", marker="o")
plt.plot(alphas, lassovalues, label="LASSO", marker="s")
plt.legend()
plt.show()

# finding best alpha value for both ridge and LASSO and printing them
best_alp_ridge = np.argmax(r2values)
best_alp_lasso = np.argmax(lassovalues)
print("Best alpha value for ridge: ", alphas[best_alp_ridge])
print("Best alpha value for LASSO: ", alphas[best_alp_lasso])

"""
Finding and explantion
The optimal value for ridge alpha would be: 100
The optimal value for LASSO alpha would be: 1

the R² score drops sharply for Lasso at alpha 50 and 
Ridge dropped at alpha 100, that's because
the regularization penalty becomes too strict, causing underfitting.
"""
