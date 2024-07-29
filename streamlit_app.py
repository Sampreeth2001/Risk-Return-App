import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split

# Streamlit app title
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Stock Price Forecaster by <a href='https://github.com/scma-632'>SCMA 632</a></h1>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <p align="center">
      <a href="https://github.com/DenverCoder1/readme-typing-svg">
        <img src="https://readme-typing-svg.herokuapp.com?font=Time+New+Roman&color=yellow&size=30&center=true&vCenter=true&width=600&height=100&lines=Stock+Forecasts+Made+Simple!;stock_analyser-1.0;" alt="Typing SVG">
      </a>
    </p>
    """,
    unsafe_allow_html=True
)

# User inputs
ticker = st.text_input('Enter stock ticker symbol', 'AAPL')
start_date = st.date_input('Start date', value=pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date', value=pd.to_datetime('today'))
forecast_horizon = st.number_input('Enter forecast horizon (days)', min_value=1, value=30)

# Fetch stock data
data = yf.download(ticker, start=start_date, end=end_date)

# Pre-process data
data.reset_index(inplace=True)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
ts_data = data['Adj Close']

# Decompose time series
decomposition = seasonal_decompose(ts_data, model='multiplicative', period=12)
st.write('Seasonal decomposition:')
st.pyplot(decomposition.plot())

# Train-test split
monthly_data = ts_data.resample("M").mean()
train, test = train_test_split(monthly_data, test_size=0.2, shuffle=False)

# Model fitting
st.write('Fitting model...')
model = auto_arima(train, seasonal=True,m=12, suppress_warnings=True)

# Generate forecast
forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)

# Plot the original data, fitted values, and forecast
plt.figure(figsize=(12, 6))
plt.plot(train, label='Original Data')
plt.plot(forecast.index, forecast, label='Forecast', color='green')
plt.fill_between(forecast.index, 
                 conf_int[:, 0], 
                 conf_int[:, 1], 
                 color='k', alpha=.15)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Auto ARIMA Forecasting')
st.pyplot(plt)

st.write('Forecasted values:')
st.write(forecast)
