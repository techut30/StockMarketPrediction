# Quantum-Enhanced Stock Prediction

A comparative analysis project exploring classical machine learning versus quantum-enhanced prediction models for stock market forecasting using PennyLane. The project includes data ingestion, data preprocessing, model training, and deployment using Docker and Kubernetes.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Ingestion](#data-ingestion)
  - [Data Preprocessing](#data-preprocessing)
  - [Classical ML Model](#classical-ml-model)
  - [Quantum-Enhanced Model](#quantum-enhanced-model)
- [Deployment](#deployment)
- [Future Scope](#future-scope)
- [References](#references)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Stock market prediction is highly challenging due to the complex, non-linear, and high-dimensional nature of financial data. This project evaluates whether integrating quantum computing techniques with classical machine learning can significantly enhance predictive accuracy. Our quantum-enhanced model, built using PennyLane, achieved 94% accuracy compared to 81% for classical models.

---

## Features

- **Data Ingestion:** Fetches historical stock data from the Alpha Vantage API and stores it as CSV files.
- **Data Preprocessing:** Cleans and enriches data by computing technical indicators (Moving Averages, RSI, MACD, etc.), applying noise reduction techniques (Savitzky–Golay filter), and generating lag features.
- **Classical ML Model:** Implements a Random Forest Regressor for benchmark predictions.
- **Quantum-Enhanced Model:** Utilizes variational quantum circuits (with RY rotations and CNOT gates) and the Adam optimizer to achieve superior prediction accuracy.
- **Deployment:** Implements a Flask API for predictions, containerized with Docker and orchestrated via Kubernetes for scalability.
- **Comparative Analysis:** Evaluation metrics include RMSE, MAE, MAPE, and overall accuracy.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/quantum-stock-prediction.git
   cd quantum-stock-prediction

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux or macOS
   venv\Scripts\activate      # For Windows

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Configure your API keys: **
   Replace YOUR_API_KEY in data_ingestion.py with your actual Alpha Vantage API key.

## Usage

### Data Ingestion
- **Script:** `data_ingestion.py`
- **Purpose:** Fetch historical stock data for multiple symbols from the Alpha Vantage API and save the data as CSV files in the `data/raw` directory.
- **How to Run:**
  ```bash
  python data_ingestion.py


## Data Preprocessing

- **Script:** `data_preprocessing.py`
- **Purpose:**
  - Load raw stock data from CSV files in the `data/raw` directory.
  - Apply noise reduction using the Savitzky–Golay filter to smooth out the closing price series.
  - Compute technical indicators such as Moving Averages (MA_10, MA_50, MA_20), RSI, MACD, and Volatility.
  - Generate lag features (e.g., `lag_1`, `lag_2`, `lag_3`) to capture temporal dependencies.
  - Combine data from multiple stock symbols into a single DataFrame and add a symbol identifier.
  - Split the enriched data into training and testing sets while preserving the sequential order.
  - Save the processed dataset and the train-test splits to the `data/processed` and `data/splits` directories respectively.
- **How to Run:**
  ```bash
  python data_preprocessing.py

## Classical ML Model

- **Script:** `classical_ML.py`
- **Purpose:**
  - Load the preprocessed dataset from `data/processed/processed_data.csv`.
  - Select key technical indicators (e.g., MA_10, MA_50, RSI, MACD, Signal_Line) as input features.
  - Use the closing price as the target variable.
  - Split the dataset into training and testing sets, maintaining the time-series order.
  - Train a Random Forest Regressor on the training set.
  - Evaluate the model using performance metrics such as RMSE, R², MAE, MAPE, and F1-score.
  - Save the trained model as a `.pkl` file in the `models/classical` directory.
  - Generate stock recommendations by simulating future returns and identifying optimal sell dates based on predicted returns.
- **How to Run:**
  ```bash
  python classical_ML.py


## Quantum-Enhanced Model

- **Script:** `quantum_model.py`
- **Purpose:**
  - Load the preprocessed dataset (features and target) from `data/processed/processed_data.csv`.
  - Normalize the features and target values to ensure numerical stability in the quantum circuit.
  - Define a variational quantum circuit using PennyLane:
    - **Data Encoding:** Use RY gates to encode normalized feature values into quantum states (via \( \theta = \arcsin(\text{clip}(x, -1, 1)) \)).
    - **Variational Layers:** Apply 4 layers where each layer performs parameterized RY rotations followed by entanglement using CNOT gates.
    - **Measurement:** Compute the expectation value of the Pauli-Z operator on the first qubit.
  - Utilize the Adam optimizer to update the circuit’s parameters over 50 epochs, minimizing a cost function that combines Mean Squared Error (MSE) and L2 regularization.
  - Generate predictions on both the test set and the full dataset, then de-normalize the predicted outputs.
  - Evaluate the model performance using metrics such as RMSE, MAE, and MAPE.
  - Save the trained quantum model parameters (weights) in the `models/quantum` directory and output evaluation metrics to a report file.
- **How to Run:**
  ```bash
  python quantum_model.py


## Deployment

- **Flask Backend:**
  - A Flask API is used to serve prediction and recommendation endpoints.
  - This API integrates both the classical and quantum models, handling incoming requests and returning predictions.

- **Docker Containerization:**
  - The entire application is containerized using Docker to ensure a consistent environment across different systems.
  - A multi-stage Dockerfile is used to separate development dependencies from the runtime environment, resulting in a lean production image.
  - Example command to build the Docker image:
    ```bash
    docker build -t quantum-stock-prediction .
    ```

## Future Scope

- **Real-Time Data Integration:**
  - Incorporate live-streaming stock data to enable continuous and dynamic forecasting.

- **Advanced Quantum Algorithms:**
  - Explore new quantum circuit designs and error mitigation strategies as quantum hardware advances.
  
- **Hybrid Quantum-Classical Models:**
  - Further integrate classical and quantum techniques to harness the strengths of both for improved prediction accuracy.

- **Expanded Feature Set:**
  - Integrate supplementary data sources (e.g., news sentiment, social media trends) to enhance model inputs and predictive power.

- **Cross-Industry Applications:**
  - Adapt the developed framework for predictive analytics in sectors such as healthcare, manufacturing, logistics, and telecommunications.

- **Cloud-Based Deployment:**
  - Scale the system via cloud platforms (AWS, GCP) with auto-scaling, monitoring, and improved resource management.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



<!-- Watermark --> <div style="position: fixed; bottom: 10px; right: 10px; font-size: 10pt; opacity: 0.5;"> Coded by Uttakarsh </div> ```

