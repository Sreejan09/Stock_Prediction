# NUMPY : adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as ny
# PANDAS : manipulating and analysing the data(Python Data Analysis)
import pandas as pd
# MATPLOT : plotting library to create static, animated and interactive visualisations with numpy extension
import matplotlib.pyplot as plt
# PANDAS_DATAREADER : access public financial data from the Internet and import it into Python as a DataFrame
from pandas_datareader import data as pdr
# KERAS : neural network Application Programming Interface (API) for Python that is tightly integrated with TensorFlow, which is used to build machine learning models.
from keras.models import load_model
# STREAMLIT : free and open-source framework to rapidly build and share beautiful machine learning and data science web apps.
import streamlit as st
from datetime import date
# provides convenient access to the Yahoo Finance API
import yfinance as yfin
# algorithmic decision making method
from sklearn.preprocessing import MinMaxScaler



# background image
# pag_im = ""
# <style>
# [data-testid='stAppViewcontainer']{
#     background-image :url ("")
#     background-size :cover;
# }
# </style>


# adding the backgroud image to the the web app
# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://c1.wallpaperflare.com/preview/898/284/844/stock-trading-monitor-desk.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# add_bg_from_url() 

# taking date from the user to start the compilation
# static way
# today = date.today()
# end = today.strftime("%Y-%m-%d")

# heading of the web app
st.title("Stock Trend Prediction")
# taking the input of the stock
user_input = st.text_input('Enter Stock Sticker', 'SBIN.NS')
# start date for the prediction
start = '2015-01-01'
end = '2023-01-01'
start = st.text_input("Enter Start Date : yyyy-mm-dd")

# end date for the stock prediction
end = st.text_input("Enter End Date : yyyy-mm-dd")
# accessing the date from the user
yfin.pdr_override()
# start = '2020-01-01'
# end = '2023-01-01'

# collecting all thee from the  user for data frame 
df= pdr.get_data_yahoo(user_input, start, end)

st.subheader('Data from '+start+' to  ' +end)

# decribing the extracted data from the  user of the given stock
st.write(df.describe())





# Visualization

# plotting the basic graph between closing price and the time
st.subheader('Closing Price vs Time Chart')
# defining the width and height of the figure
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close, 'black')
st.pyplot(fig)



# 100 days moving average

# calculating the ma of 100 suing rolling function
st.subheader('Closing Price(Black) vs Time Chart with 100 MA(Blue)')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, 'b')
plt.plot(df.Close, 'black')
st.pyplot(fig)



# 200 day ma
st.subheader('Closing Price vs Time Chart with 100MA(Blue) & 200MA(Green)')
# ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma100, 'b')
plt.plot(ma200, 'green')
plt.plot(df.Close, 'black')
st.pyplot(fig)



# 220 and 44 day ma
st.subheader('Closing Price vs Time Chart with 44MA(Blue) & 220MA(Green)')
# When Green line is Supporting Blue line - Uptrend and visa-versa
ma44 = df.Close.rolling(44).mean()
ma220 = df.Close.rolling(220).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(ma44, 'b')
plt.plot(ma220, 'green')
# plt.plot(df.Close, 'black')
st.pyplot(fig)





# Splitting the Data into Training 70% and Testing 30%
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*.7):int(len(df))])

# scaling the values between 0-1 for futher calculation minmaxScaler from sklearn
scaler = MinMaxScaler(feature_range=(0,1))

# scaling the values in a given range for the data training
data_training_array = scaler.fit_transform(data_training) 



# Load my model
# loding the model with is trained for better speed
# tenserflow library for fast numerical computing created
model1 = load_model('keras_model.h5')

# Testing Part 
past_100_days = data_training.tail(100) 
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test, y_test = ny.array(x_test), ny.array(y_test)

# using the model for the precdiction of the y axis values
y_predicted = model1.predict(x_test)
# finding the scale value
scale = scaler.scale_
scale_factor  = 1/scale[0]
# getting the original value of predicted or testing
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor


# finally plotting the predicted value
st.subheader('Predictions(Green) vs Original(Black)')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'black', label = 'Original Price')
plt.plot(y_predicted, 'green', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig2)