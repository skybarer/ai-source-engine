"""
=============================================================================
VALIDATOR.PY - Step 4: Model Evaluation & Academic Metrics
=============================================================================

PURPOSE: Compare model predictions against held-out test data.
         Compute standard forecasting metrics used in academic literature.

METRICS WE COMPUTE:
┌──────────────────────────┬──────────────────────────────────────────────┐
│ Metric                   │ What it measures                             │
├──────────────────────────┼──────────────────────────────────────────────┤
│ MAE  (Mean Abs Error)    │ Average absolute error in mentions           │
│ RMSE (Root Mean Sq Err)  │ Like MAE but penalizes big errors more      │
│ MAPE (Mean Abs % Error)  │ Error as percentage (main metric)           │
│ Accuracy = 100 - MAPE   │ How close predictions are (our TARGET >70%) │
│ Peak Timing Error        │ Days between predicted vs actual peak       │
│ Early Detection Success  │ Did we detect peak 45-60 days early?       │
│ Direction Correct        │ Did we predict the right trend direction?   │
└──────────────────────────┴──────────────────────────────────────────────┘

WHY MAPE AS PRIMARY METRIC?
  - Scale-independent: works across products with different mention levels
  - Standard in time series literature (Makridakis et al., 2018)
  - Easy to interpret: MAPE 20% means "on average, we're 20% off"
  - Accuracy = 100 - 20 = 80% → clear for non-technical audience

WHY PEAK DETECTION?
  The business value: sellers need to know WHEN a product will peak
  to source inventory 45-60 days in advance. Being off by ±7 days
  is acceptable for sourcing decisions.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from config import VALIDATION_CONFIG, RESULTS_DIR


class ModelValidator:
    """Evaluate forecasting model against held-out test data"""

    def __init__(self):
        self.config = VALIDATION_CONFIG
        self.results = []

    def calculate_metrics(self, actual, predicted):
        """
        Compute all evaluation metrics.
        
        Args:
            actual:    numpy array of true values (test set)
            predicted: numpy array of model predictions
        
        Returns:
            dict with MAE, RMSE, MAPE, Accuracy, peak detection metrics
        """
        # Ensure arrays are same length
        n = min(len(actual), len(predicted))
        actual = actual[:n]
        predicted = predicted[:n]

        # ─── Error Metrics ───
        mae  = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        accuracy = 100 - mape  # Our primary KPI

        # ─── Peak Detection ───
        actual_peak  = np.argmax(actual)    # Day with highest actual mentions
        pred_peak    = np.argmax(predicted) # Day with highest predicted mentions
        peak_error   = abs(actual_peak - pred_peak)  # Timing error in days

        # Early detection: was peak detected 45-60 days before actual?
        early_min, early_max = self.config['early_detection_window']
        early_success = early_min <= peak_error <= early_max

        # Direction: did model predict correct trend (up/down)?
        actual_direction = 'up' if actual[-1] > actual[0] else 'down'
        pred_direction   = 'up' if predicted[-1] > predicted[0] else 'down'

        return {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2),
            'Accuracy': round(accuracy, 2),
            'Peak_Timing_Error_Days': int(peak_error),
            'Early_Detection_Success': bool(early_success),
            'Direction_Correct': bool(actual_direction == pred_direction),
            'Actual_Peak_Day': int(actual_peak),
            'Predicted_Peak_Day': int(pred_peak)
        }

    def validate_single_product(self, product_df, model, product_name):
        """
        Validate model on one product using train/test split.
        
        Split strategy:
          [───── train_days ─────][── test_days ──]
          
        The model trains on the first portion and predicts the second.
        We compare prediction vs actual test data.
        """
        train_days = self.config['train_days']
        test_days = self.config['test_days']

        # Need minimum data to validate
        min_required = max(40, test_days + 20)
        if len(product_df) < min_required:
            return None

        # Adapt split to available data
        actual_train = min(train_days, len(product_df) - test_days)
        actual_test = min(test_days, len(product_df) - actual_train)
        if actual_test < 5:
            return None

        # Split
        train_df = product_df[:actual_train].copy()
        test_df = product_df[actual_train:actual_train + actual_test].copy()
        actual = test_df['mentions'].values

        # Forecast and evaluate
        try:
            forecast_result = model.ensemble_forecast(train_df, verbose=0)
            predicted = forecast_result['forecast']
            metrics = self.calculate_metrics(actual, predicted)

            return {
                'Product': product_name,
                **metrics,
                'Train_Size': len(train_df),
                'Test_Size': len(test_df),
                'Forecast_Mean': round(predicted.mean(), 2),
                'Actual_Mean': round(actual.mean(), 2)
            }
        except Exception as e:
            print(f"[WARN] Validation failed for {product_name}: {e}")
            return None

    def validate_all_products(self, df_all, model, max_products=None):
        """Validate model on multiple products and save results"""
        print("=" * 60)
        print("RUNNING MODEL VALIDATION")
        print("=" * 60)

        products = df_all['product'].unique()
        if max_products:
            products = products[:max_products]

        results = []
        for i, product in enumerate(products, 1):
            print(f"\n[{i}/{len(products)}] Validating {product}...")
            product_df = df_all[df_all['product'] == product].reset_index(drop=True)
            result = self.validate_single_product(product_df, model, product)

            if result:
                results.append(result)
                print(f"  [OK] MAPE: {result['MAPE']:.2f}% | Accuracy: {result['Accuracy']:.2f}%")
            else:
                print(f"  [SKIP] Insufficient data")

        if not results:
            print("\n[FAIL] No products could be validated")
            return None

        results_df = pd.DataFrame(results)
        results_df.to_csv(RESULTS_DIR / "validation_metrics.csv", index=False)
        print(f"\n[OK] Results saved to {RESULTS_DIR / 'validation_metrics.csv'}")
        return results_df

    def print_summary(self, results_df):
        """Print a human-readable validation summary"""
        print(f"\n{'=' * 60}")
        print(f"VALIDATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Products Validated: {len(results_df)}")
        print(f"Average MAPE: {results_df['MAPE'].mean():.2f}% (target: <{self.config['target_mape']}%)")
        print(f"Average Accuracy: {results_df['Accuracy'].mean():.2f}% (target: >70%)")
        print(f"Average Peak Error: {results_df['Peak_Timing_Error_Days'].mean():.1f} days")
        print(f"Direction Correct: {results_df['Direction_Correct'].mean() * 100:.1f}%")

        target_met = results_df['Accuracy'].mean() >= 70
        print(f"\n{'[OK] TARGET MET' if target_met else '[FAIL] TARGET NOT MET'}: "
              f"Accuracy {'>' if target_met else '<'} 70%")
        return {'avg_accuracy': results_df['Accuracy'].mean(), 'target_met': target_met}


# ==================== STANDALONE TEST ====================
if __name__ == "__main__":
    from data_loader import KaggleDataLoader
    from forecasting_model import HybridForecastingModel

    loader = KaggleDataLoader()
    df = loader.load_and_merge_all()
    model = HybridForecastingModel()
    validator = ModelValidator()
    results = validator.validate_all_products(df, model, max_products=5)
    if results is not None:
        validator.print_summary(results)
