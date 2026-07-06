import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
import numpy as np


# reading the data and splitting with ,
data = pd.read_csv("50_Startups.csv", delimiter=",")
df = pd.DataFrame(data)

# exploring the variables and datatype
print(df.info())

# excluding the state column from df because heatmap won't accept str
numeric_df = df.select_dtypes(include="number")

# using heatmap to check correlations
sns.heatmap(data=numeric_df.corr().round(2), annot=True, cmap="coolwarm")
plt.tight_layout()
plt.show()

# drawing scatter plot to see linear relationship existence
plt.subplot(1, 2, 1)
plt.scatter(df["R&D Spend"], df["Profit"])
plt.xlabel("R&D Spend")
plt.ylabel("Profit")

# doing the same thing for marketing variable as well.
plt.subplot(1, 2, 2)
plt.scatter(df["Marketing Spend"], df["Profit"])
plt.xlabel("Marketing Spend")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()

# setting values to x, y. X has two columns from df and Y has Profit.
x = df[["R&D Spend", "Marketing Spend"]]
y = df["Profit"]

# splitting the data to 80/20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# creating and training model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# predicting profit by training and test data
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# measuring accuracy of training data and test data with rmse and r2
rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
r2 = metrics.r2_score(y_train, y_train_pred)

# doing the same thing for test data
rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
r2_test = metrics.r2_score(y_test, y_test_pred)

# showing/printing the results
print("Train RMSE: ", rmse, "R2: ", r2)
print("Test RMSE: ", rmse_test, "R2: ", r2_test)

"""
3- The reason I selected r&d spend and marketing spend for the model is because they both have a strong correlation based on 
the heatmap, and administration has very weak close to 0 correlations with the profit.  so adding it(administration) variable won't have large affect on the predictions.
"""
