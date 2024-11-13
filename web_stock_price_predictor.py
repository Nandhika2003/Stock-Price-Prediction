import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# App title
st.markdown("<h1 style='text-align: center;'>📈 Stock Price Predictor App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Get historical data and future predictions for stock prices.</h3>", unsafe_allow_html=True)

# Input section for stock ID
st.sidebar.header("Stock Selection")
stock = st.sidebar.text_input("Enter the Stock ID", "GOOG")

# Date range input for stock data
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Stock data loading with spinner
with st.spinner(f"Fetching {stock} data..."):
    google_data = yf.download(stock, start, end)

# Load the pre-trained model
model = load_model("Latest_stock_price_model.keras")

# Display the stock data
st.subheader(f"Stock Data for {stock}")
st.write(google_data)

# Moving Average settings
st.sidebar.header("Moving Average Settings")
ma_100 = st.sidebar.checkbox("Show MA for 100 Days", value=True)
ma_200 = st.sidebar.checkbox("Show MA for 200 Days")
ma_250 = st.sidebar.checkbox("Show MA for 250 Days")

# Data splitting
splitting_len = int(len(google_data) * 0.7)
x_test = google_data[['Close']].iloc[splitting_len:].copy()  # Ensure 'Close' column exists in x_test

# Debugging output for x_test
st.write("Debugging Info: x_test columns:", x_test.columns)
st.write(x_test.head())

# Function to plot graphs
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None, title="Stock Price"):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'Orange', label='Moving Average')
    plt.plot(full_data.Close, 'b', label='Original Close Price')
    if extra_data:
        plt.plot(extra_dataset, 'g', label='Extra Data')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.grid(True)
    return fig

# Plots for Moving Averages
if ma_250:
    google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
    st.subheader('Original Close Price and MA for 250 days')
    st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data))

if ma_200:
    google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
    st.subheader('Original Close Price and MA for 200 days')
    st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data))

if ma_100:
    google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
    st.subheader('Original Close Price and MA for 100 days')
    st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data))

# Combined plot for 100-day and 250-day MAs
if ma_100 and ma_250:
    st.subheader('Original Close Price vs MA for 100 days and MA for 250 days')
    st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Prediction section
scaler = MinMaxScaler(feature_range=(0,1))

# Scale data if 'Close' column exists
if 'Close' in x_test.columns:
    scaled_data = scaler.fit_transform(x_test[['Close']])
else:
    st.error("Error: 'Close' column is missing in x_test.")
    scaled_data = None

# Proceed if scaled_data is not None
if scaled_data is not None:
    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)
    predictions = model.predict(x_data)

    # Inverse transform the predictions and actual values
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Dataframe to hold predictions and actual values
    ploting_data = pd.DataFrame({
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    }, index=google_data.index[splitting_len+100:])

    # Display prediction results
    st.subheader("Original vs Predicted Stock Prices")
    st.write(ploting_data)

    # Plot for predictions
    st.subheader('Original Close Price vs Predicted Close Price')
    fig = plt.figure(figsize=(15,6))
    plt.plot(pd.concat([google_data.Close[:splitting_len+100], ploting_data], axis=0))
    plt.legend(["Data - Not Used", "Original Test Data", "Predicted Test Data"])
    plt.title(f"Stock Price Prediction for {stock}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    st.pyplot(fig)
