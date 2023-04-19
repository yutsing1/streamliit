# %%
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import metrics
import pickle 
# %%
car_dataset=pd.read_csv("car_dataset_train.csv")
# %%
# checking the number of missing values
car_dataset.isnull().sum()
# %%
# getting some information about the dataset
car_dataset.info()
# %%
# checking the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Selling_type.value_counts())
print(car_dataset.Transmission.value_counts())

# %%
car_dataset.head()
# %%
#defining data for training. The inputs (regressors, 𝑥) and output (response, 𝑦).
X_train = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y_train = car_dataset['Selling_Price']

# %%
# loading the linear regression model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train,Y_train)
# %%
# prediction on Training data
training_data_prediction = lin_reg_model.predict(X_train)
# %%
# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)
# %%
#save the modle
pickle.dump(lin_reg_model,open('car_price.pkl','wb'))
# %%
