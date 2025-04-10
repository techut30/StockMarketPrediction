import os
import sys
import signal
import time
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from datetime import timedelta
import joblib

app = Flask(__name__)

def signal_handler(signum, frame):
    print("\nWarning: Termination signal received. Exiting program.")
    sys.exit(1)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

PROJECT_ROOT = "/Users/uttakarsh/Desktop/StockMarketPrediction"
MODEL_PATH_QUANTUM = os.path.join(PROJECT_ROOT, "models/quantum/quantum_model.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/processed_data.csv")

quantum_model = joblib.load(MODEL_PATH_QUANTUM)
df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

@app.route("/")
def index():
    return "Hello, this is the quantum stock prediction API!"

@app.route("/predict/quantum", methods=["POST"])
def predict_quantum():
    data = request.get_json()
    features_input = np.array(data["features"]).reshape(1, -1)
    prediction = quantum_model(features_input)
    return jsonify({"prediction": prediction.tolist()})

@app.route("/recommend/stocks", methods=["GET"])
def recommend_stocks():
    top_n = request.args.get("top_n", default=5, type=int)
    df_recommend = df.groupby('Symbol').agg({'Return_Percentage': 'mean', 'close': 'last'}).reset_index()
    df_recommend = df_recommend.sort_values(by='Return_Percentage', ascending=False)
    top_stocks = df_recommend.head(top_n).to_dict(orient="records")
    return jsonify({"top_stocks": top_stocks})

@app.route("/recommend/sell_date", methods=["GET"])
def recommend_sell_date():
    symbol = request.args.get("symbol", type=str)
    investment_amount = request.args.get("investment_amount", type=float)
    symbol_df = df[df['Symbol'] == symbol].copy()
    if symbol_df.empty:
        return jsonify({"error": f"No data found for symbol '{symbol}'"}), 404
    last_date = symbol_df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
    future_data = symbol_df.iloc[-1:].copy()
    future_data = pd.concat([future_data] * 30, ignore_index=True)
    future_data.index = future_dates
    quantum_features = ['MA_10', 'MA_50', 'RSI', 'MACD', 'Signal_Line']
    missing_features = [f for f in quantum_features if f not in future_data.columns]
    if missing_features:
        return jsonify({"error": f"Missing features in data: {missing_features}"}), 400
    future_predictions = []
    for i in range(len(future_data)):
        row_features = future_data.iloc[i][quantum_features].values.reshape(1, -1)
        pred = quantum_model(row_features)
        future_predictions.append(pred[0])
    future_data['Predicted_Close'] = future_predictions
    future_data['Return_Percentage'] = (future_data['Predicted_Close'] - future_data['close']) / future_data['close'] * 100
    best_sell_date = future_data['Return_Percentage'].idxmax()
    best_sell_price = future_data.loc[best_sell_date, 'Predicted_Close']
    buy_price = symbol_df['close'].iloc[-1]
    shares = investment_amount / buy_price
    profit = (best_sell_price - buy_price) * shares
    return jsonify({
        "symbol": symbol,
        "buy_price": buy_price,
        "best_sell_date": best_sell_date.strftime('%Y-%m-%d'),
        "best_sell_price": best_sell_price,
        "profit": profit
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
