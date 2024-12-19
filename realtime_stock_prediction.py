import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Data Retrieval
def fetch_realtime_data(ticker, start_date, end_date):
    """Retrieve stock data from Yahoo Finance."""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval="1m")
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Step 2: Feature Engineering
def prepare_features(data):
    """Generate features for the model."""
    data['Return'] = data['Adj Close'].pct_change()
    data['MA10'] = data['Adj Close'].rolling(10).mean()
    data['MA50'] = data['Adj Close'].rolling(50).mean()
    data['EMA10'] = data['Adj Close'].ewm(span=10, adjust=False).mean()
    data['EMA50'] = data['Adj Close'].ewm(span=50, adjust=False).mean()
    data['Volatility'] = data['Return'].rolling(10).std()
    data.dropna(inplace=True)
    return data

# Step 3: Train Model
def train_model(data):
    """Train a Random Forest model."""
    X = data[['MA10', 'MA50', 'EMA10', 'EMA50', 'Volatility']]
    y = data['Adj Close']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X, y

# Step 4: Make Predictions
def make_predictions(model, data):
    """Make predictions using the trained model."""
    X = data[['MA10', 'MA50', 'EMA10', 'EMA50', 'Volatility']]
    predictions = model.predict(X)
    return predictions

# Streamlit App Layout
st.title("📈 Real-Time Stock Prediction with Streamlit")
st.sidebar.title("Settings")

# User Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL").upper()
start_date = st.sidebar.date_input("Start Date:", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date:", pd.to_datetime("2023-01-01"))

# Fetch Data
if ticker:
    data = fetch_realtime_data(ticker, start_date, end_date)
    if data is not None and not data.empty:
        st.write(f"Displaying real-time stock data for **{ticker}**.")
        
        # Feature Engineering
        data = prepare_features(data)

        # Train Model
        model, X, y = train_model(data)
        
        # Predictions
        predictions = make_predictions(model, data)
        
        # Plot Actual vs Predicted
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Adj Close'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=data.index, y=predictions, mode='lines', name='Predicted'))
        fig.update_layout(
            title=f"Actual vs Predicted Prices for {ticker}",
            xaxis_title="Time",
            yaxis_title="Stock Price (USD)",
            template="plotly_white"
        )
        st.plotly_chart(fig)

        # Display Metrics
        rmse = np.sqrt(mean_squared_error(y, predictions))
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
        
        # Show Recent Data
        st.subheader("Recent Data")
        st.write(data.tail())
    else:
        st.warning("No data available. Please check the ticker symbol.")

