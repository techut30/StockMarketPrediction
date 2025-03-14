import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def load_raw_data(symbols, raw_data_path):
    dfs = []
    for symbol in symbols:
        file_path = os.path.join(raw_data_path, f"{symbol}_daily.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df['Symbol'] = symbol
            dfs.append(df)
        else:
            print(f"File not found for symbol: {symbol}")
    return pd.concat(dfs)

def create_features(df):
    if 'close' not in df.columns:
        raise ValueError("Column 'close' not found in the data.")
    df['MA_10'] = df['close'].rolling(window=10).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['close'])
    df['MACD'], df['Signal_Line'] = compute_macd(df['close'])
    df = df.dropna()
    return df

def split_data(df, test_size=0.2):
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    return train_df, test_df

def preprocess_data(symbols, raw_data_path, processed_data_path, splits_data_path):
    df = load_raw_data(symbols, raw_data_path)
    if df.empty:
        raise ValueError("No data loaded. Check the raw data files.")
    df = create_features(df)
    train_df, test_df = split_data(df)
    os.makedirs(processed_data_path, exist_ok=True)
    os.makedirs(splits_data_path, exist_ok=True)
    df.to_csv(os.path.join(processed_data_path, "processed_data.csv"))
    train_df.to_csv(os.path.join(splits_data_path, "train.csv"))
    test_df.to_csv(os.path.join(splits_data_path, "test.csv"))
    print(f"Processed data saved to {processed_data_path}")
    print(f"Train-test splits saved to {splits_data_path}")

if __name__ == '__main__':
    symbols = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "NVDA", "V", "JNJ",
        "WMT", "PYPL", "DIS", "NFLX", "INTC", "KO", "GS", "CVX", "XOM", "AMD",
        "MA", "CSCO", "PEP", "PFE", "MCD", "WFC"
    ]
    raw_data_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/data/raw"
    processed_data_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/data/processed"
    splits_data_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/data/splits"
    preprocess_data(symbols, raw_data_path, processed_data_path, splits_data_path)