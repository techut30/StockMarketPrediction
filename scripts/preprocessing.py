import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter  # For smoothing


def reduce_noise(series, window_length=11, polyorder=2):
    """
    Applies Savitzky–Golay filter to smooth the input series.
    window_length must be odd and less than or equal to the size of the series.
    """
    # Ensure window_length is odd and not greater than the series length
    if window_length >= len(series):
        window_length = len(series) - 1 if len(series) % 2 == 0 else len(series)
    if window_length % 2 == 0:
        window_length += 1
    return pd.Series(savgol_filter(series, window_length, polyorder), index=series.index)


def compute_rsi(data, window=14):
    delta = data.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
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
    if dfs:
        return pd.concat(dfs)
    else:
        return pd.DataFrame()


def create_features(df):
    # Check for required column
    if 'close' not in df.columns:
        raise ValueError("Column 'close' not found in the data.")

    # Reduce noise by smoothing the 'close' column
    df['close_smoothed'] = reduce_noise(df['close'], window_length=11, polyorder=2)

    # Compute technical indicators on the smoothed data
    df['MA_10'] = df['close_smoothed'].rolling(window=10).mean()
    df['MA_50'] = df['close_smoothed'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['close_smoothed'])
    df['MACD'], df['Signal_Line'] = compute_macd(df['close_smoothed'])

    # Optionally, add additional noise-reduced features (e.g., volatility computed on the smoothed series)
    df['Volatility'] = df['close_smoothed'].rolling(window=20).std()

    # Add a 20-day moving average using the smoothed close
    df['MA_20'] = df['close_smoothed'].rolling(window=20).mean()

    # Add lag features based on the smoothed price to help capture trends
    for lag in range(1, 4):
        df[f'lag_{lag}'] = df['close_smoothed'].shift(lag)

    # Drop rows with missing values after feature engineering
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
    # Replace with your relative or generic data directory paths
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"
    splits_data_path = "data/splits"
    preprocess_data(symbols, raw_data_path, processed_data_path, splits_data_path)
