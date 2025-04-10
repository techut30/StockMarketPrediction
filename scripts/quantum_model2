import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pennylane as qml
import pennylane.numpy as pnp
import joblib
from datetime import timedelta

np.random.seed(42)

processed_data_path = "data/processed/processed_data.csv"
models_path = "models/quantum"
reports_path = "reports/quantum_ml"
os.makedirs(models_path, exist_ok=True)
os.makedirs(reports_path, exist_ok=True)

df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)

features = ['MA_10', 'MA_50', 'RSI', 'MACD', 'Signal_Line']
target = 'close'

df['MA_5'] = df['close'].rolling(window=5).mean()
df['MA_20'] = df['close'].rolling(window=20).mean()
df['MA_100'] = df['close'].rolling(window=100).mean()
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['Volatility_5'] = df['close'].rolling(window=5).std()
df['Volatility_20'] = df['close'].rolling(window=20).std()
df['Price_Momentum_5'] = df['close'].pct_change(periods=5)
df['Price_Momentum_10'] = df['close'].pct_change(periods=10)
if 'volume' in df.columns:
    df['Volume_MA_5'] = df['volume'].rolling(window=5).mean()
    df['Volume_Change'] = df['volume'].pct_change()
    features.extend(['Volume_MA_5', 'Volume_Change'])
df['Upper_Band'] = df['MA_20'] + (df['Volatility_20'] * 2)
df['Lower_Band'] = df['MA_20'] - (df['Volatility_20'] * 2)
df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA_20']
for lag in range(1, 7):
    df[f'lag_{lag}'] = df['close'].shift(lag)
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
additional_features = ['MA_5', 'MA_20', 'MA_100', 'EMA_12', 'EMA_26',
                       'Volatility_5', 'Volatility_20', 'Price_Momentum_5', 'Price_Momentum_10',
                       'Upper_Band', 'Lower_Band', 'BB_Width', 'day_of_week', 'month']
features.extend(additional_features)
features.extend([f'lag_{lag}' for lag in range(1, 7)])
df = df.dropna()
df = df.tail(1000)

X = df[features].values.astype(np.float64)
y = df[target].values.astype(np.float64)

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)
target_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, shuffle=False)

n_qubits = 10
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits)

def pad_input(x, n):
    if len(x) < n:
        pad = np.zeros(n - len(x))
        return np.concatenate([x, pad])
    return x[:n]

@qml.qnode(dev, interface='autograd')
def quantum_circuit(inputs, weights):
    x_pad = pad_input(inputs, n_qubits)
    for i in range(n_qubits):
        val = pnp.clip(x_pad[i], -1, 1)
        qml.RY(pnp.arcsin(val), wires=i)
        qml.RZ(pnp.arccos(val**2), wires=i)
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RX(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)
        if layer % 2 == 0:
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        else:
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

def quantum_model(inputs, weights):
    return quantum_circuit(inputs, weights)

def cost(weights, X, y, alpha=0.05):
    batch_size = min(32, len(X))
    indices = np.random.choice(len(X), batch_size, replace=False)
    X_batch = X[indices]
    y_batch = y[indices]
    preds = pnp.array([quantum_model(x, weights) for x in X_batch])
    mse = pnp.mean((preds - y_batch) ** 2)
    reg = alpha * pnp.sum(weights ** 2)
    return mse + reg

weights_shape = (n_layers, n_qubits, 3)
theta_init = 0.01 * np.random.randn(*weights_shape)
params = pnp.array(theta_init, requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.01)
n_epochs = 300

def lr_schedule(epoch):
    return 0.01 * (0.95 ** (epoch // 30))

best_cost = float('inf')
patience = 20
no_improvement = 0

for epoch in range(n_epochs):
    opt.stepsize = lr_schedule(epoch)
    params = opt.step(lambda w: cost(w, X_train, y_train), params)
    if epoch % 10 == 0:
        current_cost = cost(params, X_train, y_train)
        print(f"Epoch {epoch}: Cost = {current_cost}")
        if current_cost < best_cost:
            best_cost = current_cost
            no_improvement = 0
            best_weights = params.copy()
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                params = best_weights
                break

weights = params
model_file = os.path.join(models_path, "quantum_model_advanced.pkl")
joblib.dump(weights, model_file)
print(f"Model saved to {model_file}")

def ensemble_predict(x, weights, n_ensemble=5):
    predictions = []
    for _ in range(n_ensemble):
        x_noisy = x + np.random.normal(0, 0.01, x.shape)
        pred = quantum_model(x_noisy, weights)
        predictions.append(pred)
    return np.mean(predictions)

y_pred_scaled = np.array([ensemble_predict(x, weights) for x in X_test])
y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
r2 = r2_score(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)
mape = mean_absolute_percentage_error(y_test_actual, y_pred)
accuracy = 100 - (mape * 100)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"Accuracy: {accuracy:.2f}%")

metrics_file = os.path.join(reports_path, "quantum_model_advanced_metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R²: {r2}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"MAPE: {mape}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
print(f"Metrics saved to {metrics_file}")

all_pred_scaled = np.array([ensemble_predict(x, weights) for x in X_scaled])
all_pred = target_scaler.inverse_transform(all_pred_scaled.reshape(-1, 1)).flatten()
df['Predicted_Close'] = all_pred
df['Return_Percentage'] = (df['Predicted_Close'] - df['close']) / df['close'] * 100

def recommend_stocks(df, top_n=5):
    df_recommend = df.groupby('Symbol').agg({
        'Return_Percentage': 'mean', 
        'close': 'last',
        'Volatility_20': 'last',
        'RSI': 'last'
    }).reset_index()
    df_recommend['Score'] = df_recommend['Return_Percentage'] * 0.6 + -df_recommend['Volatility_20'] * 0.2 + (1 - abs(df_recommend['RSI'] - 50) / 50) * 0.2
    df_recommend = df_recommend.sort_values(by='Score', ascending=False)
    return df_recommend.head(top_n)

top_stocks = recommend_stocks(df)
print("\nTop 5 Stocks to Invest In:")
print(top_stocks)
recommendations_file = os.path.join(reports_path, "stock_recommendations_advanced.csv")
top_stocks.to_csv(recommendations_file, index=False)
print(f"Recommendations saved to {recommendations_file}")

def recommend_sell_date(symbol, investment_amount):
    symbol_df = df[df['Symbol'] == symbol].copy()
    last_date = symbol_df.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
    n_simulations = 20
    all_scenarios = []
    for _ in range(n_simulations):
        future_data = symbol_df.iloc[-1:].copy()
        future_data = pd.concat([future_data] * 30, ignore_index=True)
        future_data.index = future_dates
        for feature in features:
            if feature in future_data.columns:
                base_value = future_data[feature].iloc[0]
                noise = np.random.normal(0, 0.01 * np.arange(1, 31), 30) * base_value
                future_data[feature] = base_value + noise
        future_features = feature_scaler.transform(future_data[features].values)
        future_preds_scaled = np.array([quantum_model(x, weights) for x in future_features])
        future_preds = target_scaler.inverse_transform(future_preds_scaled.reshape(-1, 1)).flatten()
        future_data['Predicted_Close'] = future_preds
        future_data['Return_Percentage'] = (future_data['Predicted_Close'] - symbol_df['close'].iloc[-1]) / symbol_df['close'].iloc[-1] * 100
        all_scenarios.append(future_data['Predicted_Close'])
    future_predictions = pd.DataFrame(all_scenarios).transpose()
    future_predictions.index = future_dates
    future_summary = pd.DataFrame({
        'Mean_Prediction': future_predictions.mean(axis=1),
        'Lower_CI': future_predictions.quantile(0.05, axis=1),
        'Upper_CI': future_predictions.quantile(0.95, axis=1)
    })
    best_sell_date = future_summary['Mean_Prediction'].idxmax()
    best_sell_price = future_summary.loc[best_sell_date, 'Mean_Prediction']
    buy_price = symbol_df['close'].iloc[-1]
    shares = investment_amount / buy_price
    profit = (best_sell_price - buy_price) * shares
    return best_sell_date, best_sell_price, profit, future_summary

investment_amount = float(input("Enter your investment amount: $"))
print("\nInvestment Recommendations:")
for symbol in top_stocks['Symbol']:
    sell_date, sell_price, profit, summary = recommend_sell_date(symbol, investment_amount)
    print(f"\nStock: {symbol}")
    print(f"Buy Today at: ${df[df['Symbol'] == symbol]['close'].iloc[-1]:.2f}")
    print(f"Sell on: {sell_date.strftime('%Y-%m-%d')} at: ${sell_price:.2f}")
    print(f"Expected Profit: ${profit:.2f}")
