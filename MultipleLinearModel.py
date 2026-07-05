import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model, metrics
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# loading built in database
data = load_diabetes(as_frame=True)
df = data.frame

# drawing the heat map to see correlations between futures
sns.heatmap(data=df.corr().round(2),  annot=True, cmap="coolwarm")
plt.show()

# dividing the figure into 3 section to show/compare scatter plot of the futures
plt.subplot(1, 3, 1)
plt.scatter(df["bp"], df["target"])
plt.xlabel("bp")
plt.ylabel("target")

plt.subplot(1, 3, 2)
plt.scatter(df["s5"], df["target"])
plt.xlabel("s5")
plt.ylabel("target")

plt.subplot(1, 3, 3)
plt.scatter(df["bmi"], df["target"])
plt.xlabel("bmi")
plt.ylabel("target")
plt.show()

# defining x to bmi,s5,bp values which we get from df, and y to target
x = pd.DataFrame(df[["bmi", "s5", "bp"]], columns=["bmi", "s5", "bp"])
y = df["target"]

# declearing variables and splitting the dataset for training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=5)

# creating model
model = linear_model.LinearRegression()
model.fit(x_train, y_train) # training the model

# predicting y by the model
y_train_predict = model.predict(x_train)
# scoring rmse and r2 for the predictions
rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_predict))
r2 = metrics.r2_score(y_train, y_train_predict)

# doing the same thing as above for the remaining 20% test datas
y_test_predict = model.predict(x_test)
rmse_test = np.sqrt(metrics.mean_squared_error(y_test, y_test_predict))
r2_test = metrics.r2_score(y_test, y_test_predict)

print("\nModel with bmi, s5 and bp")
print("Train RMSE: ", rmse, "R2: ", r2)
print("Test RMSE: ", rmse_test, "R2: ", r2_test)

# making the second model for comparison
# we go through exactly the steps as mentioned above
x1 = pd.DataFrame(df[["bmi", "s5"]], columns=["bmi", "s5"])
y1 = df["target"]

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=5)
model2 = linear_model.LinearRegression()
model2.fit(x1_train, y1_train)

y1_train_predict = model2.predict(x1_train)
rmse_train_m2 = np.sqrt(metrics.mean_squared_error(y1_train, y1_train_predict))
r2_train_m2 = metrics.r2_score(y1_train, y1_train_predict)

y1_test_predict = model2.predict(x1_test)
rmse_test_m2 = np.sqrt(metrics.mean_squared_error(y1_test, y1_test_predict))
r2_test_m2 = metrics.r2_score(y1_test, y1_test_predict)

print("\nModel 2 bmi and s5")
print("Train RMSE: ", rmse_train_m2, "R2: ", r2_train_m2)
print("Test RMSE: ", rmse_test_m2, "R2: ", r2_test_m2)

# creating model 3
x2 = pd.DataFrame(df[["bmi", "bp", "s3", "s5"]], columns=["bmi", "bp", "s3", "s5"])
y2 = df["target"]

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=5)

model3 = linear_model.LinearRegression()
model3.fit(x2_train, y2_train)

y2_train_predict = model3.predict(x2_train)
rmse2_train_m3 = np.sqrt(metrics.mean_squared_error(y2_train, y2_train_predict))
r2_train_m3 = metrics.r2_score(y2_train, y2_train_predict)

y2_test_predict = model3.predict(x2_test)
rmse2_test_m3 = np.sqrt(metrics.mean_squared_error(y2_test, y2_test_predict))
r2_test_m3 = metrics.r2_score(y2_test, y2_test_predict)

print("\nModel3 adding more features(bmi, bp, s3, s5)")
print("Train RMSE: ", rmse2_train_m3, "R2: ", r2_train_m3)
print("Test RMSE: ", rmse2_test_m3, "R2: ", r2_test_m3)
