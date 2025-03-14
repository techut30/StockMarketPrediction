import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, f1_score
import joblib
from datetime import timedelta

processed_data_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/data/processed/processed_data.csv"
splits_data_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/data/splits"
models_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/models/classical"
reports_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/reports/classical_ml"

os.makedirs(models_path, exist_ok=True)
os.makedirs(reports_path, exist_ok=True)

df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)

features = ['MA_10', 'MA_50', 'RSI', 'MACD', 'Signal_Line']
target = 'close'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

model_file = os.path.join(models_path, "random_forest_model.pkl")
joblib.dump(model, model_file)
print(f"Model saved to {model_file}")

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
accuracy = 100 - (mape * 100)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"Accuracy: {accuracy:.2f}%")

y_test_dir = (y_test > y_test.shift(1)).astype(int).iloc[1:]
y_pred_series = pd.Series(y_pred, index=y_test.index)
y_pred_dir = (y_pred_series > y_test.shift(1)).astype(int).iloc[1:]
f1 = f1_score(y_test_dir, y_pred_dir)
print(f"F1 Score: {f1}")

metrics_file = os.path.join(reports_path, "classical_model_metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R²: {r2}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"MAPE: {mape}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
    f.write(f"F1 Score: {f1}\n")
print(f"Metrics saved to {metrics_file}")

df['Predicted_Close'] = model.predict(X)
df['Return_Percentage'] = (df['Predicted_Close'] - df['close']) / df['close'] * 100

def recommend_stocks(df, top_n=5):
    df_recommend = df.groupby('Symbol').agg({'Return_Percentage': 'mean', 'close': 'last'}).reset_index()
    df_recommend = df_recommend.sort_values(by='Return_Percentage', ascending=False)
    return df_recommend.head(top_n)

top_stocks = recommend_stocks(df)
print("\nTop 5 Stocks to Invest In:")
print(top_stocks)
recommendations_file = os.path.join(reports_path, "stock_recommendations.csv")
top_stocks.to_csv(recommendations_file, index=False)
print(f"Recommendations saved to {recommendations_file}")

def recommend_sell_date(symbol, investment_amount):
    symbol_df = df[df['Symbol'] == symbol].copy()
    last_date = symbol_df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
    future_data = symbol_df.iloc[-1:].copy()
    future_data = pd.concat([future_data] * 30, ignore_index=True)
    future_data.index = future_dates
    future_data['Predicted_Close'] = model.predict(future_data[features])
    future_data['Return_Percentage'] = (future_data['Predicted_Close'] - future_data['close']) / future_data['close'] * 100
    best_sell_date = future_data['Return_Percentage'].idxmax()
    best_sell_price = future_data.loc[best_sell_date, 'Predicted_Close']
    buy_price = symbol_df['close'].iloc[-1]
    shares = investment_amount / buy_price
    profit = (best_sell_price - buy_price) * shares
    return best_sell_date, best_sell_price, profit

investment_amount = float(input("Enter your investment amount: $"))

print("\nInvestment Recommendations:")
for symbol in top_stocks['Symbol']:
    sell_date, sell_price, profit = recommend_sell_date(symbol, investment_amount)
    print(f"\nStock: {symbol}")
    print(f"Buy Today at: ${df[df['Symbol'] == symbol]['close'].iloc[-1]:.2f}")
    print(f"Sell on: {sell_date.strftime('%Y-%m-%d')} at: ${sell_price:.2f}")
    print(f"Expected Profit: ${profit:.2f}")
