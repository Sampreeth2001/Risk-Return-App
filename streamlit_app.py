import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Function to download data from yfinance
def download_data(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=3*365)
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to train the random forest model
def train_model(data):
    data = data[['Adj Close']]
    data['Date'] = data.index
    data['Date'] = data['Date'].map(datetime.toordinal)
    X = data[['Date']]
    y = data['Adj Close']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Function to evaluate the model
def evaluate_model(model, data):
    data = data[['Adj Close']]
    data['Date'] = data.index
    data['Date'] = data['Date'].map(datetime.toordinal)
    X = data[['Date']]
    y = data['Adj Close']
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    return mse

# Function to predict future trend
def predict_future(model, days=30):
    future_dates = [datetime.today() + timedelta(days=i) for i in range(1, days+1)]
    future_dates_ordinal = [date.toordinal() for date in future_dates]
    future_predictions = model.predict(np.array(future_dates_ordinal).reshape(-1, 1))
    future_data = pd.DataFrame({'Date': future_dates, 'Predicted Adj Close': future_predictions})
    return future_data

# Streamlit app
st.title('Share Risk and Return Predictor')

ticker = st.text_input('Enter the ticker of the share:', 'AAPL')

if ticker:
    data = download_data(ticker)
    st.write(f"Downloaded data for {ticker}")
    st.line_chart(data['Adj Close'])

    model = train_model(data)
    mse = evaluate_model(model, data)
    st.write(f'Model Mean Squared Error: {mse}')

    future_data = predict_future(model)
    st.write('Future Trend Prediction:')
    st.line_chart(future_data.set_index('Date'))
    st.write('Predicted data:', future_data)
