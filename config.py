# config.py

# API Credentials
api_key = "PKS19REHC9DNFTSL79J9"
secret_key = "eyhObNiTzKiEXQiSYevhlJEXk5NhUdSfyS3z6EQu"

# Environment settings
paper = True

# Alpaca URLs (These might need to be set depending on the environment)
trade_api_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
trade_api_wss = "wss://paper-api.alpaca.markets/stream" if paper else "wss://api.alpaca.markets/stream"
data_api_url = "https://data.alpaca.markets"
stream_data_wss = "wss://data.alpaca.markets/stream"

# Tickers da usare
tickers = ["BTC/USD", "ETH/USD"]