# AI-Multiple-Linear-Regression-
Creating Multiple linear regression models with python

**--- Diebetes Model ---**

Firstable we start with only two independant variables bmi and s5 and the result for that model is:
Model 2 bmi and s5
Train RMSE:  56.560890965481114 R2:  0.4507519215172524
Test RMSE:  57.1759740950605 R2:  0.4815610845742896

then I would add bp future to the second model, because it has strong correlation with the target and it is not multicollinear with others.
<img width="647" height="559" alt="heatmap" src="https://github.com/user-attachments/assets/eba8cff8-5611-4400-8eb6-116b3fc2344d" />
<img width="903" height="549" alt="scatterPlot" src="https://github.com/user-attachments/assets/7bcd6130-061b-4645-bab3-332489d674cf" />
When we compare the two models together the result is: 


Model with bmi, s5 and bp
Train RMSE:  55.32610611166316 R2:  0.47447150038132146
Test RMSE:  56.6256100515053 R2:  0.4914938186648421

Model 2 bmi and s5
Train RMSE:  56.560890965481114 R2:  0.4507519215172524
Test RMSE:  57.1759740950605 R2:  0.4815610845742896

This shows that adding bp(blood pressure) to the model gives us larger R2(training and testing) which indicates more correct predictions which is good and also gives us smaller RMSE(training and testing) which also means smaller errors averaglly. 

If we test the theory of adding even more futurest to the model for example adding s3 to the model and this was the result of that:

Model3 adding more features(bmi, bp, s3, s5)
Train RMSE:  54.62693258577694 R2:  0.48767011273913297
Test RMSE:  56.294220259918674 R2:  0.4974282593239341

If i compare this to (Model with bmi, s5 and bp) what i see is very slightly improvements in the training data and almost no improvement in test data, because s3 and s5 have high correllation as shown in heatmap so there is multicollinear and it doesn't actually imporve our model by adding more futures at least s3 doesn't improve it.

**--- Startup Model ---**

In this model I start by loading the data to dataframe and exploring what is on it by _.info()_ <img width="643" height="553" alt="heatmap" src="https://github.com/user-attachments/assets/f6e18c9c-c2c3-4471-9263-854406759356" /> then we draw a heatmap of that dataset so we can see correlation of the variables.

The reason I selected _r&d spend_ and _marketing spend_ for the model is because they both have a strong correlation based on 
the heatmap, and administration has very weak close to 0 correlations with the profit.  so adding it(administration) variable won't have large affect on the predictions.


after that we draw a scatter plot of the variables <img width="643" height="547" alt="subplot" src="https://github.com/user-attachments/assets/00848b7c-81e5-4923-9e1f-67f44c43e37e" /> and the target so we can see the linear regression relationship between them, then I split the dataset 80/20 for training and testing the models and create the models then we use the models to make predictions of profit for the startups.
<img width="547" height="435" alt="output" src="https://github.com/user-attachments/assets/16e3ccd9-505e-4c67-b7ea-8cd7c7cda15f" />
