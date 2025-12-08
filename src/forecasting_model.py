"""
Hybrid Forecasting Model: LSTM + ARIMA + Prophet Ensemble
Target: 60-day forecast with >70% accuracy
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

from config import MODEL_CONFIG, LSTM_MODEL_PATH, SCALER_PATH


class HybridForecastingModel:
    """
    Ensemble forecasting model combining:
    - LSTM (50% weight) - Non-linear patterns
    - ARIMA (30% weight) - Linear time series
    - Prophet (20% weight) - Seasonality
    """

    def __init__(self):
        self.config = MODEL_CONFIG
        self.lookback = self.config['lookback_days']
        self.horizon = self.config['forecast_horizon']

        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.arima_order = self.config['arima_order']

        # Ensemble weights
        self.weights = {
            'lstm': 0.5,
            'arima': 0.3,
            'prophet': 0.2
        }

    # ==================== LSTM Model ====================

    def build_lstm_model(self):
        """Build LSTM architecture"""
        units = self.config['lstm_units']

        model = Sequential([
            LSTM(units[0], return_sequences=True, input_shape=(self.lookback, 1)),
            Dropout(self.config['dropout_rate']),
            LSTM(units[1], return_sequences=True),
            Dropout(self.config['dropout_rate']),
            LSTM(units[2], return_sequences=False),
            Dense(64, activation='relu'),
            Dropout(self.config['dropout_rate']),
            Dense(self.horizon)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        return model

    def prepare_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []

        for i in range(len(data) - self.lookback - self.horizon):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.horizon])

        return np.array(X), np.array(y)

    def train_lstm(self, series, verbose=0):
        """
        Train LSTM model on time series

        Args:
            series: pandas Series of mentions/sales data
            verbose: Training verbosity (0=silent, 1=progress)

        Returns:
            Trained model
        """
        # Scale data
        scaled_data = self.scaler.fit_transform(series.values.reshape(-1, 1))

        # Prepare sequences
        X, y = self.prepare_sequences(scaled_data)

        if len(X) == 0:
            print("⚠️  Not enough data for LSTM training")
            return None

        # Reshape for LSTM [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Build and train
        self.lstm_model = self.build_lstm_model()

        early_stop = EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )

        self.lstm_model.fit(
            X, y,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            verbose=verbose,
            callbacks=[early_stop],
            validation_split=0.1
        )

        return self.lstm_model

    def predict_lstm(self, series):
        """Generate LSTM forecast"""
        if self.lstm_model is None:
            return np.zeros(self.horizon)

        # Scale last lookback window
        scaled_data = self.scaler.transform(
            series.values[-self.lookback:].reshape(-1, 1)
        )

        # Reshape for prediction
        X = scaled_data.reshape((1, self.lookback, 1))

        # Predict and inverse transform
        prediction_scaled = self.lstm_model.predict(X, verbose=0)
        prediction = self.scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        ).flatten()

        return np.maximum(prediction, 0)  # No negative values

    # ==================== ARIMA Model ====================

    def predict_arima(self, series):
        """
        Generate ARIMA forecast

        Args:
            series: pandas Series of data

        Returns:
            numpy array of predictions
        """
        try:
            model = ARIMA(series, order=self.arima_order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=self.horizon)
            return np.maximum(forecast.values, 0)

        except Exception as e:
            print(f"⚠️  ARIMA failed: {e}")
            return np.zeros(self.horizon)

    # ==================== Prophet Model ====================

    def predict_prophet(self, df):
        """
        Generate Prophet forecast

        Args:
            df: DataFrame with 'date' and 'mentions' columns

        Returns:
            numpy array of predictions
        """
        try:
            # Prepare data for Prophet
            prophet_df = df[['date', 'mentions']].rename(
                columns={'date': 'ds', 'mentions': 'y'}
            )

            # Build model
            model = Prophet(
                seasonality_mode=self.config['prophet_seasonality'],
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95
            )

            # Suppress Prophet logs
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)

            model.fit(prophet_df)

            # Forecast
            future = model.make_future_dataframe(periods=self.horizon)
            forecast = model.predict(future)

            return np.maximum(
                forecast['yhat'].tail(self.horizon).values,
                0
            )

        except Exception as e:
            print(f"⚠️  Prophet failed: {e}")
            return np.zeros(self.horizon)

    # ==================== Ensemble Prediction ====================

    def ensemble_forecast(self, df, verbose=0):
        """
        Generate weighted ensemble forecast

        Args:
            df: DataFrame with 'date' and 'mentions' columns
            verbose: Print progress (0=silent, 1=verbose)

        Returns:
            dict with forecast, confidence intervals, and components
        """
        if verbose:
            print(f"Training ensemble for {df['product'].iloc[0] if 'product' in df.columns else 'product'}...")

        series = df['mentions']

        # Train LSTM
        if verbose:
            print("  - Training LSTM...")
        self.train_lstm(series, verbose=0)
        lstm_pred = self.predict_lstm(series)

        # ARIMA prediction
        if verbose:
            print("  - Fitting ARIMA...")
        arima_pred = self.predict_arima(series)

        # Prophet prediction
        if verbose:
            print("  - Fitting Prophet...")
        prophet_pred = self.predict_prophet(df)

        # Weighted ensemble
        ensemble = (
                self.weights['lstm'] * lstm_pred +
                self.weights['arima'] * arima_pred +
                self.weights['prophet'] * prophet_pred
        )

        # Calculate confidence intervals (using LSTM variance as proxy)
        std = np.std(lstm_pred) * 1.5
        lower_bound = ensemble - 1.96 * std
        upper_bound = ensemble + 1.96 * std

        if verbose:
            print("  ✓ Ensemble forecast complete")

        return {
            'forecast': ensemble,
            'lower_bound': np.maximum(lower_bound, 0),
            'upper_bound': upper_bound,
            'components': {
                'lstm': lstm_pred,
                'arima': arima_pred,
                'prophet': prophet_pred
            }
        }

    # ==================== Model Persistence ====================

    def save_model(self, path=None):
        """Save trained LSTM model and scaler"""
        if path is None:
            path = LSTM_MODEL_PATH

        if self.lstm_model is not None:
            self.lstm_model.save(path)
            joblib.dump(self.scaler, SCALER_PATH)
            print(f"✓ Model saved to {path}")

    def load_model(self, path=None):
        """Load trained LSTM model and scaler"""
        if path is None:
            path = LSTM_MODEL_PATH

        try:
            self.lstm_model = load_model(path)
            self.scaler = joblib.load(SCALER_PATH)
            print(f"✓ Model loaded from {path}")
            return True
        except:
            print(f"⚠️  Could not load model from {path}")
            return False


# Test the model
if __name__ == "__main__":
    from data_loader import KaggleDataLoader

    print("Testing Hybrid Forecasting Model...")

    # Load data
    loader = KaggleDataLoader()
    df = loader.load_and_merge_all()

    # Get one product for testing
    product = df['product'].unique()[0]
    product_df = df[df['product'] == product].copy()

    print(f"\nTesting on: {product}")
    print(f"Data points: {len(product_df)}")

    # Use first 120 days for training
    train_df = product_df[:120].copy()

    # Initialize model
    model = HybridForecastingModel()

    # Generate forecast
    print("\nGenerating 60-day forecast...")
    forecast = model.ensemble_forecast(train_df, verbose=1)

    print("\n" + "=" * 60)
    print("FORECAST RESULTS")
    print("=" * 60)
    print(f"Forecast mean: {forecast['forecast'].mean():.2f}")
    print(f"Forecast min: {forecast['forecast'].min():.2f}")
    print(f"Forecast max: {forecast['forecast'].max():.2f}")
    print(f"95% CI width: {(forecast['upper_bound'] - forecast['lower_bound']).mean():.2f}")

    # Save model
    model.save_model()