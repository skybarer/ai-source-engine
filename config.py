"""
Configuration settings for AI Trend-to-Source Engine
"""

import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Paths
MODELS_DIR = PROJECT_ROOT / "models"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model.h5"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Output Paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                  MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model Hyperparameters
MODEL_CONFIG = {
    "lookback_days": 30,           # Historical window for LSTM
    "forecast_horizon": 60,        # Predict 60 days ahead
    "lstm_units": [128, 64, 32],   # LSTM layer sizes
    "dropout_rate": 0.2,
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.001,
    "arima_order": (2, 1, 2),      # (p, d, q)
    "prophet_seasonality": "multiplicative"
}

# Trend Scoring Weights
TREND_SCORING = {
    "growth_velocity_weight": 0.4,
    "sentiment_weight": 0.2,
    "saturation_weight": 0.2,
    "profit_weight": 0.2,
    "high_potential_threshold": 60,  # Score > 60 triggers alert
    "velocity_threshold": 5           # Score velocity > 5
}

# Validation Settings
VALIDATION_CONFIG = {
    "train_days": 30,      # ↓ Reduced to work with sparse per-product data
    "test_days": 15,       # ↓ Reduced for flexible time series windows
    "min_products_for_validation": 3,  # ↓ More lenient threshold
    "target_mape": 30.0,              # Target < 30% MAPE (70% accuracy)
    "early_detection_window": (45, 60) # Days before peak
}

# Kaggle Dataset Paths (update these after downloading)
KAGGLE_DATA = {
    "amazon": RAW_DATA_DIR / "amazon_sales.csv",
    "flipkart": RAW_DATA_DIR / "flipkart_products.csv",
    # "reddit": RAW_DATA_DIR / "reddit_comments.csv"
}

# Categories to Analyze
PRODUCT_CATEGORIES = [
    "Fashion",
    "Electronics",
    "Home & Kitchen",
    "Beauty",
    "Lifestyle Accessories"
]

print(f"[OK] Configuration loaded. Project root: {PROJECT_ROOT}")