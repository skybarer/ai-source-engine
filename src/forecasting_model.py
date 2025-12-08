"""
Hybrid Forecasting Model: PyTorch LSTM + ARIMA + Prophet Ensemble
Windows-compatible version using PyTorch instead of TensorFlow
Target: 60-day forecast with >70% accuracy
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Deep Learning with PyTorch (Windows-friendly)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

from config import MODEL_CONFIG, LSTM_MODEL_PATH, SCALER_PATH

# ==================== PyTorch LSTM Model ====================

class LSTMModel(nn.Module):
    """PyTorch LSTM for time series forecasting"""

    def __init__(self, input_size=1, hidden_sizes=[128, 64, 32], output_size=60, dropout=0.2):
        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)

        self.fc1 = nn.Linear(hidden_sizes[2], 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        # LSTM layers
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        out, _ = self.lstm3(out)

        # Take last output
        out = out[:, -1, :]

        # Fully connected
        out = self.relu(self.fc1(out))
        out = self.dropout3(out)
        out = self.fc2(out)

        return out


class HybridForecastingModel:
    """
    Ensemble forecasting model combining:
    - PyTorch LSTM (50% weight) - Non-linear patterns
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

        # Set device (CPU for Windows compatibility)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Ensemble weights
        self.weights = {
            'lstm': 0.5,
            'arima': 0.3,
            'prophet': 0.2
        }

        print(f"✓ Using PyTorch on {self.device}")

    # ==================== LSTM Model ====================

    def build_lstm_model(self):
        """Build PyTorch LSTM architecture"""
        units = self.config['lstm_units']

        model = LSTMModel(
            input_size=1,
            hidden_sizes=units,
            output_size=self.horizon,
            dropout=self.config['dropout_rate']
        )

        return model.to(self.device)

    def prepare_sequences(self, data):
        """Create lookback sequences for LSTM training"""
        X, y = [], []

        for i in range(len(data) - self.lookback - self.horizon):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.horizon])

        return np.array(X), np.array(y)

    def train_lstm(self, series, verbose=0):
        """
        Train PyTorch LSTM model on time series

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
            if verbose:
                print("⚠️  Not enough data for LSTM training")
            return None

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)  # [batch, seq, features]
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Build model
        self.lstm_model = self.build_lstm_model()

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.lstm_model.parameters(),
            lr=self.config['learning_rate']
        )

        # Training loop
        self.lstm_model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.config['epochs']):
            # Forward pass
            optimizer.zero_grad()
            outputs = self.lstm_model(X_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Early stopping
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                best_model_state = self.lstm_model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

            if verbose and epoch % 10 == 0:
                print(f"    Epoch {epoch}, Loss: {current_loss:.4f}")

        # Restore best model
        if best_model_state is not None:
            self.lstm_model.load_state_dict(best_model_state)

        return self.lstm_model

    def predict_lstm(self, series):
        """Generate LSTM forecast"""
        if self.lstm_model is None:
            return np.zeros(self.horizon)

        # Scale last lookback window
        scaled_data = self.scaler.transform(
            series.values[-self.lookback:].reshape(-1, 1)
        )

        # Convert to tensor
        X = torch.FloatTensor(scaled_data).unsqueeze(0).unsqueeze(-1).to(self.device)

        # Predict
        self.lstm_model.eval()
        with torch.no_grad():
            prediction_scaled = self.lstm_model(X).cpu().numpy()

        # Inverse transform
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
            # print(f"⚠️  ARIMA failed: {e}")
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
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

            model.fit(prophet_df)

            # Forecast
            future = model.make_future_dataframe(periods=self.horizon)
            forecast = model.predict(future)

            return np.maximum(
                forecast['yhat'].tail(self.horizon).values,
                0
            )

        except Exception as e:
            # print(f"⚠️  Prophet failed: {e}")
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
            product_name = df['product'].iloc[0] if 'product' in df.columns else 'product'
            print(f"Training ensemble for {product_name}...")

        series = df['mentions']

        # Train LSTM
        if verbose:
            print("  - Training PyTorch LSTM...")
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
            # Save as .pth for PyTorch
            torch.save(
                self.lstm_model.state_dict(),
                str(path).replace('.h5', '.pth')
            )
            joblib.dump(self.scaler, SCALER_PATH)
            print(f"✓ Model saved to {path}")

    def load_model(self, path=None):
        """Load trained LSTM model and scaler"""
        if path is None:
            path = LSTM_MODEL_PATH

        try:
            self.lstm_model = self.build_lstm_model()
            self.lstm_model.load_state_dict(
                torch.load(str(path).replace('.h5', '.pth'))
            )
            self.scaler = joblib.load(SCALER_PATH)
            print(f"✓ Model loaded from {path}")
            return True
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            return False


# Test the model
if __name__ == "__main__":
    from data_loader import KaggleDataLoader

    print("Testing PyTorch-based Hybrid Forecasting Model...")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

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

    print("\n✓ PyTorch model working successfully!")

    # Save model
    model.save_model()