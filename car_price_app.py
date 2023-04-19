# %%
import pickle
import streamlit as st
import pandas as pd
from sklearn import metrics
from sklearn.metrics import mean_squared_error
# %%
# Locate the model and pick our model as "regressor"
pickle_a=open("/Users/yuqingyang/Desktop/streamliit/5_pickle/car_price.pkl","rb")
regressor=pickle.load(pickle_a) 
# %%
st.title("Linear Regression App")
st.header("Upload a CSV file for prediction")
uploaded_file = st.file_uploader("Choose a file", type=["csv"])
# %%
if uploaded_file is not None:
    # Read the uploaded file as a Pandas DataFrame
    df = pd.read_csv(uploaded_file)
# %%
if st.button("Predict"):
        X = df.drop(['Car_Name','Selling_Price'],axis=1)
        Y = df['Selling_Price']
        training_data_prediction = regressor.predict(X) #result will be displayed if button is pressed
        mse = mean_squared_error(Y, training_data_prediction)
        st.success("Mean Squared Error : {}".format(mse))

