import os
import sys
import signal
import time
import threading
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy

stop_timer = False

class TimerThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.start_time = time.time()
    def run(self):
        while not stop_timer:
            elapsed = time.time() - self.start_time
            print(f"Elapsed time: {elapsed:.2f} seconds", end='\r')
            time.sleep(1)

def signal_handler(signum, frame):
    print("\nWarning: Termination signal received. Exiting program.")
    sys.exit(1)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

timer_thread = TimerThread()
timer_thread.start()

np.random.seed(42)
tf.random.set_seed(42)

processed_data_path = "data/processed/processed_data.csv"
models_path = "models/quantum"
reports_path = "reports/quantum_ml"
os.makedirs(models_path, exist_ok=True)
os.makedirs(reports_path, exist_ok=True)

df = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)

features = ['MA_10', 'MA_50', 'RSI', 'MACD', 'Signal_Line']
target = 'close'
df['MA_20'] = df['close'].rolling(window=20).mean()
df['Volatility'] = df['close'].rolling(window=20).std()
features.extend(['MA_20', 'Volatility'])
for lag in range(1, 4):
    df[f'lag_{lag}'] = df['close'].shift(lag)
features.extend([f'lag_{lag}' for lag in range(1, 4)])
df = df.dropna()
df = df.tail(500).reset_index(drop=True)

n_qubits = len(features)
qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]

def create_quantum_circuit(sample):
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.rx(sample[i])(qubit))
    for i in range(n_qubits - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit

scaler = StandardScaler()
X_all = df[features].values.astype(np.float32)
X_scaled = scaler.fit_transform(X_all)
y_all = df[target].values.astype(np.float32).reshape(-1, 1)
target_scaler = StandardScaler()
y_scaled = target_scaler.fit_transform(y_all)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

circuits_train = [create_quantum_circuit(sample) for sample in X_train]
circuits_test = [create_quantum_circuit(sample) for sample in X_test]

quantum_data_train = tfq.convert_to_tensor(circuits_train)
quantum_data_test = tfq.convert_to_tensor(circuits_test)

symbols = sympy.symbols('theta(0:%d)' % n_qubits)
def create_model_circuit():
    circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        circuit.append(cirq.ry(symbols[i])(qubit))
    return circuit

model_circuit = create_model_circuit()
observable = cirq.Z(qubits[0])
quantum_layer = tfq.layers.PQC(model_circuit, operators=observable)

inputs = tf.keras.Input(shape=(), dtype=tf.string)
x = quantum_layer(inputs)
outputs = tf.keras.layers.Dense(1)(x)
hybrid_model = tf.keras.Model(inputs=inputs, outputs=outputs)

hybrid_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                     loss='mse',
                     metrics=['mae'])
hybrid_model.summary()

history = hybrid_model.fit(quantum_data_train, y_train,
                           validation_data=(quantum_data_test, y_test),
                           epochs=20, batch_size=16)

loss, mae_metric = hybrid_model.evaluate(quantum_data_test, y_test)
print("\nTest Loss (MSE):", loss)
print("Test MAE:", mae_metric)

y_pred_scaled = hybrid_model.predict(quantum_data_test)
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_test_denorm = target_scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_denorm, y_pred))
mae = mean_absolute_error(y_test_denorm, y_pred)
mape = mean_absolute_percentage_error(y_test_denorm, y_pred)
accuracy = 100 - (mape * 100)
r2 = r2_score(y_test_denorm, y_pred)

print(f"\nRMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"Accuracy: {accuracy:.2f}%")
print(f"R²: {r2}")

metrics_file = os.path.join(reports_path, "hybrid_model_metrics.txt")
with open(metrics_file, "w") as f:
    f.write(f"RMSE: {rmse}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"MAPE: {mape}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
    f.write(f"R²: {r2}\n")
print(f"Metrics saved to {metrics_file}")

circuits_all = [create_quantum_circuit(sample) for sample in X_scaled]
quantum_data_all = tfq.convert_to_tensor(circuits_all)
all_pred_scaled = hybrid_model.predict(quantum_data_all)
all_pred = target_scaler.inverse_transform(all_pred_scaled)
df['Predicted_Close'] = all_pred.flatten()
df['Return_Percentage'] = (df['Predicted_Close'] - df['close']) / df['close'] * 100

def recommend_stocks(dataframe, top_n=3):
    df_recommend = dataframe.groupby('Symbol').agg({'Return_Percentage': 'mean', 'close': 'last'}).reset_index()
    df_recommend = df_recommend.sort_values(by='Return_Percentage', ascending=False)
    return df_recommend.head(top_n)

top_stocks = recommend_stocks(df)
print("\nTop 3 Stocks to Invest In:")
print(top_stocks)
recommendations_file = os.path.join(reports_path, "stock_recommendations_tfq.csv")
top_stocks.to_csv(recommendations_file, index=False)
print(f"Recommendations saved to {recommendations_file}")

def recommend_sell_date(symbol, investment_amount):
    symbol_df = df[df['Symbol'] == symbol].copy()
    last_date = symbol_df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    future_data = symbol_df.iloc[-1:].copy()
    future_data = pd.concat([future_data] * 30, ignore_index=True)
    future_data.index = future_dates
    X_future = future_data[features].values.astype(np.float32)
    X_future_scaled = scaler.transform(X_future)
    circuits_future = [create_quantum_circuit(sample) for sample in X_future_scaled]
    quantum_data_future = tfq.convert_to_tensor(circuits_future)
    future_pred_scaled = hybrid_model.predict(quantum_data_future)
    future_data['Predicted_Close'] = target_scaler.inverse_transform(future_pred_scaled).flatten()
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

stop_timer = True
timer_thread.join()
