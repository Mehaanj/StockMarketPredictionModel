import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Download historical stock data for Apple (AAPL) using yfinance
ticker = '^BSESN'  # Change this to any stock ticker you'd like (e.g., TSLA for Tesla)
data = yf.download(ticker, start='2020-10-01', end='2025-3-29') # Change the Date

# Check if data is fetched correctly
if data.empty:
    print("No data fetched from Yahoo Finance. Check ticker or internet connection.")


data = data[['Close']].dropna()
data['Days'] = np.arange(len(data))  # Create a 'Days' feature for regression

# Split the data into training and testing sets
X = data[['Days']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 3. Build Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make Predictions
data['Predicted'] = model.predict(X)

# Forecast future values for the next 10 days, 1 month (30 days), and 1 year (365 days)
future_days = np.array([[len(data) + i] for i in range(1, 366)])  # Future days
future_predictions = model.predict(future_days)

# 5. Plot historical data and predictions
plt.figure(figsize=(12, 8))
plt.plot(data.index, data['Close'], label=f'Historical {ticker} Close Price', alpha=0.5)
plt.plot(data.index, data['Predicted'], label='Predicted Price', color='orange')
plt.plot(pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=365), future_predictions, label='Future Predictions', color='green')

plt.title(f'{ticker} Price Prediction Using Linear Regression')
plt.xlabel('Date')
plt.ylabel(f'{ticker} Close Price')
plt.legend()
plt.show()

# 6. Buy/Sell Recommendation based on the forecast trend
def buy_sell_recommendation(predictions):
    if predictions[-1] > predictions[0]:
        return "BUY"
    else:
        return "SELL"

# Recommendations for future predictions
recommendation_10_days = buy_sell_recommendation(future_predictions[:10])
recommendation_1_month = buy_sell_recommendation(future_predictions[:30])
recommendation_1_year = buy_sell_recommendation(future_predictions)

print(f"10-Day Forecast (first 10 days): {future_predictions[:10]}")
print(f"1-Month Forecast (first 30 days): {future_predictions[:30]}")
print(f"1-Year Forecast (first 365 days): {future_predictions}")

print(f"Recommendation for the next 10 days: {recommendation_10_days}")
print(f"Recommendation for the next 1 month: {recommendation_1_month}")
print(f"Recommendation for the next 1 year: {recommendation_1_year}")
