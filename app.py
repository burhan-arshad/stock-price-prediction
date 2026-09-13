import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AAL Stock Price Predictor",
    layout="wide"
)

TICKER = "AAL"
MODEL_PATH = "stock_lstm_model.keras"
SCALER_PATH = "stock_scaler.pkl"
WINDOW_PATH = "stock_window_size.pkl"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_window_size():
    return joblib.load(WINDOW_PATH)


@st.cache_data(ttl=300)
def get_stock_data(ticker):
    data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError("No stock data was returned from Yahoo Finance.")

    return data


def get_close_prices(data):
    close = data["Close"]

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.dropna()


def predict_next_price(close_prices, model, scaler, window_size):
    if len(close_prices) < window_size:
        raise ValueError(
            f"At least {window_size} trading days are required."
        )

    prices = close_prices.values.reshape(-1, 1)
    scaled_prices = scaler.transform(prices)

    last_window = scaled_prices[-window_size:]
    X_input = last_window.reshape(1, window_size, 1)

    prediction_scaled = model.predict(
        X_input,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction_scaled
    )

    return float(prediction[0][0])


try:
    model = load_model()
    scaler = load_scaler()
    window_size = load_window_size()

except Exception as e:
    st.error("Could not load the model or preprocessing files.")
    st.exception(e)
    st.stop()


try:
    stock_data = get_stock_data(TICKER)
    close_prices = get_close_prices(stock_data)

except Exception as e:
    st.error("Unable to fetch AAL stock data from Yahoo Finance.")
    st.exception(e)
    st.stop()


try:
    predicted_price = predict_next_price(
        close_prices,
        model,
        scaler,
        window_size
    )

except Exception as e:
    st.error("Prediction failed.")
    st.exception(e)
    st.stop()


last_price = float(close_prices.iloc[-1])

price_difference = predicted_price - last_price

percentage_change = (
    price_difference / last_price
) * 100

last_date = close_prices.index[-1]


st.title("AAL Stock Price Predictor")

st.write(
    f"Latest available closing data: {last_date.strftime('%Y-%m-%d')}"
)


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Last Closing Price",
        f"${last_price:.2f}"
    )

with col2:
    st.metric(
        "Predicted Next Close",
        f"${predicted_price:.2f}",
        f"${price_difference:+.2f}"
    )

with col3:
    st.metric(
        "Expected Change",
        f"{percentage_change:+.2f}%"
    )


st.subheader("Next Trading Day Prediction")


chart_prices = close_prices.tail(window_size)

next_date = chart_prices.index[-1] + pd.Timedelta(days=1)

while next_date.weekday() >= 5:
    next_date += pd.Timedelta(days=1)


fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    chart_prices.index,
    chart_prices.values,
    label="Actual Closing Price"
)

ax.scatter(
    next_date,
    predicted_price,
    s=100,
    label="Predicted Next Close"
)

ax.plot(
    [
        chart_prices.index[-1],
        next_date
    ],
    [
        last_price,
        predicted_price
    ],
    linestyle="--"
)

ax.set_title(
    "AAL — Last 60 Trading Days + Next Close Prediction"
)

ax.set_xlabel("Date")
ax.set_ylabel("Price ($)")

ax.legend()
ax.grid(alpha=0.2)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


st.subheader("Prediction Details")

detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    st.write(f"Last close: **${last_price:.2f}**")
    st.write(f"Predicted next close: **${predicted_price:.2f}**")

with detail_col2:
    st.write(f"Expected change: **{percentage_change:+.2f}%**")
    st.write(f"Sequence used: **{window_size} trading days**")


st.subheader("Recent Market Data")

latest_data = stock_data.tail(10).copy()

if isinstance(latest_data.columns, pd.MultiIndex):
    latest_data.columns = latest_data.columns.get_level_values(0)

display_columns = [
    column
    for column in ["Open", "High", "Low", "Close", "Volume"]
    if column in latest_data.columns
]

st.dataframe(
    latest_data[display_columns],
    use_container_width=True
)


with st.expander("Model Information"):
    st.write("Architecture")

    st.code(
        """
LSTM(50, return_sequences=True)
Dropout(0.5)
LSTM(50)
Dropout(0.5)
Dense(25, activation='relu')
Dense(1)
        """
    )

    st.write(f"Input sequence: {window_size} trading days")
    st.write("Output: Next closing price")
    st.write("Optimizer: Adam")
    st.write("Loss: Mean Squared Error")


st.warning(
    "This prediction is based on historical price patterns and is not "
    "financial advice or a guaranteed future price."
)


st.caption(
    "Data source: Yahoo Finance"
)