import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Set up the Streamlit app
st.title("Real-Time Stock Market Analysis and Prediction")
st.sidebar.header("Input Parameters")

# Sidebar Inputs
ticker = st.sidebar.text_input("Stock Ticker Symbol (e.g., AAPL, TSLA):", value="AAPL")
lookback_days = st.sidebar.slider("Number of Lookback Days:", min_value=30, max_value=365, value=200)
interval = st.sidebar.slider("Prediction Refresh Interval (seconds):", min_value=10, max_value=300, value=60)

# Function to fetch stock data
def fetch_stock_data(ticker, lookback_days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to prepare features
def prepare_features(data):
    data['Return'] = data['Adj Close'].pct_change()
    data['MA10'] = data['Adj Close'].rolling(10).mean()
    data['MA50'] = data['Adj Close'].rolling(50).mean()
    data['EMA10'] = data['Adj Close'].ewm(span=10, adjust=False).mean()
    data['EMA50'] = data['Adj Close'].ewm(span=50, adjust=False).mean()
    data['Volatility'] = data['Return'].rolling(10).std()
    data.dropna(inplace=True)
    return data

# Function to train model
def train_model(data):
    X = data[['MA10', 'MA50', 'EMA10', 'EMA50', 'Volatility']]
    y = data['Adj Close']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Fetch and process data
st.write(f"Fetching stock data for **{ticker}**...")
data = fetch_stock_data(ticker, lookback_days)

if not data.empty:
    data = prepare_features(data)
    st.write(f"Data preview for **{ticker}**:")
    st.dataframe(data.tail())

    # Train the model
    st.write("Training the model...")
    model = train_model(data)

    # Make predictions
    st.write("Making predictions...")
    latest_features = data[['MA10', 'MA50', 'EMA10', 'EMA50', 'Volatility']].iloc[-1:]
    prediction = model.predict(latest_features)
    current_price = data['Adj Close'].iloc[-1]
    predicted_price = prediction[0]

    # Display results
    st.write(f"### Current Price of {ticker}: ${current_price:.2f}")
    st.write(f"### Predicted Price of {ticker}: ${predicted_price:.2f}")

    # Real-time plotting
    st.write("Price Trends:")
    st.line_chart(data[['Adj Close', 'MA10', 'MA50']])

else:
    st.write(f"No data found for ticker **{ticker}**. Please check the symbol and try again.")
