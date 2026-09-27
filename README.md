# AAL Stock Price Prediction using LSTM

A deep learning project that uses an LSTM neural network to analyze historical AAL (American Airlines Group) closing prices and predict the next trading day's closing price.

## Project Overview

Stock prices are sequential data, meaning the order of previous observations matters.

In this project, the model receives the previous 60 trading days of AAL closing prices and learns patterns from the sequence to estimate the next closing price.

The project demonstrates how RNN/LSTM networks can be applied to numerical sequential data rather than only text.

## Features

* Historical stock data fetched using Yahoo Finance
* LSTM-based time-series prediction
* 60 trading-day input window
* MinMaxScaler preprocessing
* Actual vs predicted price evaluation
* RMSE and MAE evaluation
* Next trading day prediction
* Streamlit web application
* Saved trained model and scaler

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* yfinance
* Streamlit
* Joblib

## Model Architecture

```text
Input
  ↓
LSTM(50, return_sequences=True)
  ↓
Dropout(0.5)
  ↓
LSTM(50)
  ↓
Dropout(0.5)
  ↓
Dense(25, ReLU)
  ↓
Dense(1)
  ↓
Predicted Closing Price
```

## Dataset

The project does not use a manually downloaded dataset.

Historical AAL stock data is retrieved from Yahoo Finance using `yfinance`.

The model uses closing prices as its input feature.

## Data Processing

The closing prices are normalized using MinMaxScaler.

A sequence length of 60 trading days is used.

For example:

```text
Previous 60 trading days
          ↓
        LSTM
          ↓
Next trading day's closing price
```

## Training

The model is trained using:

* Optimizer: Adam
* Loss: Mean Squared Error
* Batch Size: 32
* Epochs: 50
* Sequence Length: 60

## Evaluation

The model is evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)

The evaluation compares the model's predicted closing prices with the actual closing prices from the test period.

## Streamlit Application

The Streamlit application fetches the latest AAL market data and uses the trained LSTM model to estimate the next trading day's closing price.

The application displays:

* Latest closing price
* Predicted next closing price
* Expected percentage change
* Last 60 trading days
* Next-close prediction on the chart
* Recent market data
* Model architecture

## Project Structure

```text
stock-price-prediction/
│
├── app.py
├── stock_lstm_model.keras
├── stock_scaler.pkl
├── stock_window_size.pkl
├── stock_prediction.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/burhan-arshad/stock-price-prediction
cd stock-price-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Important Note

This model was trained specifically on AAL historical stock data. It should not be used to predict other stocks without retraining the model for those stocks.

Stock prices are highly unpredictable. The model's prediction is an estimate based on historical patterns and should not be considered financial advice or a guaranteed future price.

## Future Improvements

* Train using multiple stock features such as Open, High, Low, Volume and technical indicators
* Add validation-based training instead of using the test set during training
* Prevent data leakage through train-only scaler fitting
* Compare LSTM with GRU and other forecasting approaches
* Add multiple ticker support with ticker-specific models
* Deploy the application online

```
```
