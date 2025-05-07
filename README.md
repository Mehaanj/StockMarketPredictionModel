# 📈 Stock Price Forecasting & Recommendation using Linear Regression

This project demonstrates a basic stock price forecasting system using **linear regression** on historical stock/index data fetched from Yahoo Finance using the `yfinance` API. The goal is to visualize trends, predict future closing prices, and provide simple **buy/sell recommendations** based on the predicted movement of prices.

It is intended for **educational purposes** and beginners in **machine learning**, **finance**, or **data analysis**, who want to learn how regression can be used to model time-based financial data.

⚠️ **Note**: This model assumes a linear trend in stock prices, which is rarely true in real markets. It does not account for volatility, economic factors, or technical indicators. Therefore, it should **not be used for actual trading or investment decisions**.

---

## 📦 Dependencies

Install all required Python libraries using:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn
