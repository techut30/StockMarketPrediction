import os
import requests
import pandas as pd

API_KEY = "9MIRNF77SRB2DVN3"
BASE_URL = "https://www.alphavantage.co/query"

# List of 50 major stock symbols
symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "BRK.A", "NVDA", "V", "JNJ",
    "WMT", "PYPL", "BABA", "DIS", "NFLX", "BA", "INTC", "KO", "GS", "CVX",
    "XOM", "AMD", "MA", "CSCO", "PEP", "PFE", "MCD", "WFC", "UNH", "ABT",
    "IBM", "GE", "INTU", "T", "CVS", "ATVI", "NEE", "CAT", "HD", "ADBE",
    "VZ", "ORCL", "LMT", "MMM", "SPGI", "RTX", "SQ", "MDT", "AXP", "USB",
    "HSBC", "TM", "SBUX", "COP", "HUM"
]

def get_stock_data(symbol, function="TIME_SERIES_DAILY", outputsize="compact"):
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": outputsize
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for {symbol}")
        return None

def process_stock_data(data):
    if "Time Series (Daily)" in data:
        ts = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.columns = [col.split(". ")[1] for col in df.columns]
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    else:
        print("Data format not as expected.")
        return None

def ingest_stock_data(symbol, output_path, function="TIME_SERIES_DAILY", outputsize="compact"):
    data = get_stock_data(symbol, function=function, outputsize=outputsize)
    if data:
        df = process_stock_data(data)
        if df is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path)
            print(f"Stock data for {symbol} saved to {output_path}")
        else:
            print(f"Processing failed for {symbol}.")
    else:
        print(f"Data retrieval failed for {symbol}.")

if __name__ == '__main__':
    for symbol in symbols:
        output_path = f"/Users/uttakarsh/Desktop/StockMarketPrediction/data/raw/{symbol}_daily.csv"
        ingest_stock_data(symbol, output_path)