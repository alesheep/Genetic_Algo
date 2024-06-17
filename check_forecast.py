import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from config import tickers

def load_formatted_data(ticker):
    data_dir = os.path.join(os.path.dirname(__file__), 'DATA')
    file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_close_prices_test.csv")
    return prepare_lstm_data(file_path)

def load_and_prepare_data(file_path, look_back=60):
    data_df = pd.read_csv(file_path)
    data = data_df.filter(['close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(look_back, len(train_data)):
        x_train.append(train_data[i-look_back:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = scaled_data[training_data_len - look_back:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(look_back, len(test_data)):
        x_test.append(test_data[i-look_back:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, scaler

def build_lstm_model(input_shape, units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def verify_model_performance(model, x_test, y_test, scaler):
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    
    return predictions, y_test

if __name__ == "__main__":
    ticker = tickers[0]  # Use the first ticker for this example
    # Assume we have a new recent data file 'recent_data.csv' for verification
    recent_data_file = 'path_to_recent_data/recent_data.csv'
    
    look_back = 60
    units = 16  # From the best individual
    dropout_rate = 0.374  # From the best individual
    epochs = 86  # From the best individual

    x_train, y_train, x_test, y_test, scaler = load_and_prepare_data(recent_data_file, look_back)
    
    model = build_lstm_model((x_train.shape[1], 1), units=units, dropout_rate=dropout_rate)
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=1)
    
    predictions, actuals = verify_model_performance(model, x_test, y_test, scaler)

    # Optionally, plot the results for visual verification
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 5))
    plt.plot(actuals, color='blue', label='Actual Prices')
    plt.plot(predictions, color='red', label='Predicted Prices')
    plt.title(f'{ticker} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
