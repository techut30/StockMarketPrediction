import os
import pandas as pd
import numpy as onp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import pennylane as qml
import pennylane.numpy as pnp
import joblib
from datetime import timedelta

processed_data_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/data/processed/processed_data.csv"
splits_data_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/data/splits"
models_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/models/quantum"
reports_path = "/Users/uttakarsh/Desktop/StockMarketPrediction/reports/quantum_ml"
os.makedirs(models_path, exist_ok=True)
os.makedirs(reports_path, exist_ok=True)

df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)

# Feature engineering
features = ['MA_10', 'MA_50', 'RSI', 'MACD', 'Signal_Line']
target = 'close'
df['MA_20'] = df['close'].rolling(window=20).mean()
df['Volatility'] = df['close'].rolling(window=20).std()
features.extend(['MA_20', 'Volatility'])
for lag in range(1, 4):
    df[f'lag_{lag}'] = df['close'].shift(lag)
features.extend([f'lag_{lag}' for lag in range(1, 4)])
df = df.dropna()

# Use a subset of data for efficiency
df = df.tail(500)

X = df[features].values.astype(onp.float64)
y = df[target].values.astype(onp.float64)

eps = 1e-8
X = (X - onp.mean(X, axis=0)) / (onp.std(X, axis=0) + eps)
y_mean = onp.mean(y)
y_std = onp.std(y) + eps
y_norm = (y - y_mean) / y_std

X_train, X_test, y_train, y_test = train_test_split(X, y_norm, test_size=0.2, shuffle=False)

# Quantum circuit parameters
n_qubits = 10
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

def pad_input(x, n):
    if len(x) < n:
        pad = onp.zeros(n - len(x))
        return onp.concatenate([x, pad])
    return x[:n]

@qml.qnode(dev, interface='autograd')
def quantum_circuit(inputs, weights):
    x_pad = pad_input(inputs, n_qubits)
    for i in range(n_qubits):
        val = pnp.clip(x_pad[i], -1, 1)
        qml.RY(pnp.arcsin(val), wires=i)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
    return qml.expval(qml.PauliZ(0))

def quantum_model(inputs, weights):
    return quantum_circuit(inputs, weights)

def cost(weights, X, y, alpha=0.01):
    preds = pnp.array([quantum_model(x, weights) for x in X])
    mse = pnp.mean((preds - y) ** 2)
    reg = alpha * pnp.sum(weights ** 2)
    return mse + reg

theta_init = 0.01 * onp.random.randn(n_layers, n_qubits)
params = pnp.array(theta_init, requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.1)
for epoch in range(50):
    params = opt.step(lambda w: cost(w, X_train, y_train), params)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Cost = {cost(params, X_train, y_train)}")

weights = params
model_file = os.path.join(models_path, "quantum_model.pkl")
joblib.dump(weights, model_file)
print(f"Model saved to {model_file}")

# Sequential prediction on the test set
y_pred_norm = onp.array([quantum_model(x, weights) for x in X_test])
y_pred = y_pred_norm * y_std + y_mean
y_test_denorm = y_test * y_std + y_mean
rmse = onp.sqrt(mean_squared_error(y_test_denorm, y_pred))
r2 = r2_score(y_test_denorm, y_pred)
mae = mean_absolute_error(y_test_denorm, y_pred)
mape = mean_absolute_percentage_error(y_test_denorm, y_pred)
accuracy = 100 - (mape * 100)
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"Accuracy: {accuracy:.2f}%")
metrics_file = os.path.join(reports_path, "quantum_model_metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R²: {r2}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"MAPE: {mape}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
print(f"Metrics saved to {metrics_file}")

# Sequential full dataset prediction
all_pred = onp.array([quantum_model(x, weights) for x in X])
df['Predicted_Close'] = all_pred * y_std + y_mean
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
    future_data['Predicted_Close'] = onp.array([quantum_model(x, weights) for x in future_data[features].values])
    future_data['Predicted_Close'] = future_data['Predicted_Close'] * y_std + y_mean
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


