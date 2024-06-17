from datetime import datetime
from zoneinfo import ZoneInfo
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import os
from config import api_key, secret_key, tickers

# Initialize the client
client = CryptoHistoricalDataClient(api_key, secret_key)

# Define the request parameters
start_date = datetime(2022, 7, 1, tzinfo=ZoneInfo('UTC'))
end_date = datetime(2023, 7, 7, tzinfo=ZoneInfo('UTC'))

# Function to save bars to CSV
def save_data_to_csv(data_df, ticker):
    # Create the 'DATA' directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), 'DATA')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Construct the filename with the ticker prefix
    filename = f"{ticker.replace('/', '_')}_crypto_bars_data.csv"
    # Save the dataframe to CSV in the 'DATA' directory
    data_df.to_csv(os.path.join(data_dir, filename), index=True)
    print(f"Data saved to {os.path.join(data_dir, filename)}")

# Fetch and save the data for each ticker separately
for ticker in tickers:
    request_params = CryptoBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    # Fetch the crypto bars data
    data = client.get_crypto_bars(request_params)
    # Save the data to CSV
    save_data_to_csv(data.df, ticker)
