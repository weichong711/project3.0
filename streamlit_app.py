import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import streamlit as st

# Step 1: Fetch Live Stock Data
def fetch_live_stock_data(ticker, period="6mo", interval="1d"):
    """Retrieve live stock data using yfinance."""
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError("No data found for the ticker.")
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
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

# Step 3: Train-Test Split
def split_data(data):
    """Split data into training and testing sets."""
    X = data[['MA10', 'MA50', 'EMA10', 'EMA50', 'Volatility']]
    y = data['Adj Close']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Step 5: Evaluate Model
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2, predictions

# Step 6: Visualization
def visualize_predictions(data, predictions):
    """Visualize actual vs predicted stock prices using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data['Adj Close'], name="Actual Prices", line=dict(color='blue')))
    fig.add_trace(go.Scatter(y=predictions, name="Predicted Prices", line=dict(color='orange')))
    fig.update_layout(
        title="Stock Price Prediction",
        xaxis_title="Time",
        yaxis_title="Price",
        legend=dict(x=0, y=1),
        template="plotly_dark"
    )
    return fig

# Streamlit App
def main():
    st.title("Live Stock Prediction System")
    
    ticker = st.text_input("Enter the stock ticker (e.g., AAPL for Apple):", "AAPL").upper()
    period = st.selectbox("Select data period:", ["1mo", "3mo", "6mo", "1y", "5y"])
    interval = st.selectbox("Select data interval:", ["1d", "1wk", "1mo"])
    
    if st.button("Fetch and Predict"):
        data = fetch_live_stock_data(ticker, period, interval)
        if data is not None:
            st.write(f"Fetched data for {ticker}:")
            st.dataframe(data.tail())
            
            # Prepare features and split data
            data = prepare_features(data)
            X_train, X_test, y_train, y_test = split_data(data)
            
            # Train and evaluate model
            model = train_model(X_train, y_train)
            rmse, mae, r2, predictions = evaluate_model(model, X_test, y_test)
            
            # Display metrics
            st.subheader("Model Performance")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"MAE: {mae:.2f}")
            st.write(f"R² Score: {r2:.2f}")
            
            # Visualize predictions
            fig = visualize_predictions(data, predictions)
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
 
