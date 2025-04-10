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

2. **Create a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux or macOS
   venv\Scripts\activate      # For Windows

3. **Install Dependencies
   ```bash
   pip install -r requirements.txt


