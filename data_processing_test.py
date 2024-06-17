import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from config import tickers

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

def save_close_prices_to_csv(data_df, ticker):
    try:
        # Check if the necessary columns are present
        required_columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
        if not all(column in data_df.columns for column in required_columns):
            print(f"Data for {ticker} does not have the required columns.")
            return

        # Create the 'DATA' directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), 'DATA')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")

        # Extract the 'close' column
        close_prices = data_df[['timestamp', 'close']].copy()

        # Construct the filename with the ticker prefix
        filename = f"{ticker.replace('/', '_')}_close_prices_test.csv"
        file_path = os.path.join(data_dir, filename)

        # Save the close prices to CSV in the 'DATA' directory
        close_prices.to_csv(file_path, index=False)
        print(f"Close prices test saved to {file_path}")

    except Exception as e:
        print(f"Error saving close prices test for {ticker}: {e}")

def process_and_save_close_prices():
    data_dir = os.path.join(os.path.dirname(__file__), 'DATA')
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_crypto_bars_data_test.csv")
        if os.path.exists(file_path):
            data_df = load_data(file_path)
            if data_df is not None:
                save_close_prices_to_csv(data_df, ticker)
        else:
            print(f"Data file for {ticker} not found: {file_path}")

def prepare_lstm_data(file_path):
    data_df = load_data(file_path)
    if data_df is None:
        return None, None, None

    data = data_df.filter(['close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler

def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'DATA')
    for ticker in tickers:
        file_path = os.path.join(data_dir, f"{ticker.replace('/', '_')}_close_prices_test.csv")
        if os.path.exists(file_path):
            x_train, y_train, scaler = prepare_lstm_data(file_path)
            if x_train is not None:
                print(f"Test data for {ticker} prepared for LSTM model.")
            else:
                print(f"Failed to prepare test data for {ticker}.")
        else:
            print(f"Close prices test file for {ticker} not found: {file_path}")

if __name__ == "__main__":
    process_and_save_close_prices()
    main()
