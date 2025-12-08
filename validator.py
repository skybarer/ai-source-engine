"""
Model Validation & Metrics Calculation
Validates forecasting model and calculates academic metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from config import VALIDATION_CONFIG, RESULTS_DIR


class ModelValidator:
    """Validate forecasting model and calculate metrics"""

    def __init__(self):
        self.config = VALIDATION_CONFIG
        self.results = []

    def calculate_metrics(self, actual, predicted):
        """
        Calculate comprehensive forecasting metrics

        Args:
            actual: Array of actual values
            predicted: Array of predicted values

        Returns:
            dict of metrics
        """
        # Ensure same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

        # Basic metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted) * 100

        # Accuracy (inverse of MAPE)
        accuracy = 100 - mape

        # Peak detection metrics
        actual_peak_idx = np.argmax(actual)
        pred_peak_idx = np.argmax(predicted)
        peak_timing_error = abs(actual_peak_idx - pred_peak_idx)

        # Early detection success (45-60 day window)
        early_detection_min, early_detection_max = self.config['early_detection_window']
        early_detection_success = (
                early_detection_min <= peak_timing_error <= early_detection_max
        )

        # Directional accuracy (trend direction correct?)
        actual_trend = 'up' if actual[-1] > actual[0] else 'down'
        pred_trend = 'up' if predicted[-1] > predicted[0] else 'down'
        direction_correct = actual_trend == pred_trend

        return {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2),
            'Accuracy': round(accuracy, 2),
            'Peak_Timing_Error_Days': int(peak_timing_error),
            'Early_Detection_Success': bool(early_detection_success),
            'Direction_Correct': bool(direction_correct),
            'Actual_Peak_Day': int(actual_peak_idx),
            'Predicted_Peak_Day': int(pred_peak_idx)
        }

    def validate_single_product(self, product_df, model, product_name):
        """
        Validate model on a single product

        Args:
            product_df: DataFrame with product data
            model: Trained forecasting model
            product_name: Name of product

        Returns:
            dict of validation results
        """
        train_days = self.config['train_days']
        test_days = self.config['test_days']

        # Check if enough data
        if len(product_df) < train_days + test_days:
            return None

        # Split data
        train_df = product_df[:train_days].copy()
        test_df = product_df[train_days:train_days + test_days].copy()

        # Get actual values
        actual = test_df['mentions'].values

        # Generate forecast
        try:
            forecast_result = model.ensemble_forecast(train_df, verbose=0)
            predicted = forecast_result['forecast']

            # Calculate metrics
            metrics = self.calculate_metrics(actual, predicted)

            # Add product info
            result = {
                'Product': product_name,
                **metrics,
                'Train_Size': len(train_df),
                'Test_Size': len(test_df),
                'Forecast_Mean': round(predicted.mean(), 2),
                'Actual_Mean': round(actual.mean(), 2)
            }

            return result

        except Exception as e:
            print(f"⚠️  Validation failed for {product_name}: {e}")
            return None

    def validate_all_products(self, df_all, model, max_products=None):
        """
        Run validation on multiple products

        Args:
            df_all: DataFrame with all products
            model: Forecasting model
            max_products: Limit number of products to validate

        Returns:
            DataFrame with validation results
        """
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
                print(f"  ✓ MAPE: {result['MAPE']:.2f}% | Accuracy: {result['Accuracy']:.2f}%")
                print(
                    f"  ✓ Peak Error: {result['Peak_Timing_Error_Days']} days | Early Detection: {result['Early_Detection_Success']}")
            else:
                print(f"  ✗ Skipped (insufficient data)")

        if not results:
            print("\n❌ No products could be validated")
            return None

        results_df = pd.DataFrame(results)

        # Save results
        output_path = RESULTS_DIR / "validation_metrics.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to {output_path}")

        return results_df

    def print_summary(self, results_df):
        """Print validation summary statistics"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        print(f"\nProducts Validated: {len(results_df)}")
        print(f"\nAverage Metrics:")
        print(f"  MAPE: {results_df['MAPE'].mean():.2f}% (Target: <{self.config['target_mape']}%)")
        print(f"  Accuracy: {results_df['Accuracy'].mean():.2f}% (Target: >70%)")
        print(f"  RMSE: {results_df['RMSE'].mean():.2f}")
        print(f"  MAE: {results_df['MAE'].mean():.2f}")

        print(f"\nPeak Detection:")
        print(f"  Average Timing Error: {results_df['Peak_Timing_Error_Days'].mean():.1f} days")
        print(f"  Early Detection Success Rate: {results_df['Early_Detection_Success'].mean() * 100:.1f}%")

        print(f"\nDirection Prediction:")
        print(f"  Correct Trend Direction: {results_df['Direction_Correct'].mean() * 100:.1f}%")

        # Check if target met
        target_met = results_df['Accuracy'].mean() >= 70
        print(f"\n{'✓' if target_met else '✗'} Target >70% Accuracy: {'ACHIEVED' if target_met else 'NOT MET'}")

        # Best and worst performers
        print(f"\n" + "=" * 60)
        print("TOP 3 BEST PREDICTIONS")
        print("=" * 60)
        best = results_df.nsmallest(3, 'MAPE')[['Product', 'MAPE', 'Accuracy', 'Peak_Timing_Error_Days']]
        print(best.to_string(index=False))

        print(f"\n" + "=" * 60)
        print("TOP 3 WORST PREDICTIONS")
        print("=" * 60)
        worst = results_df.nlargest(3, 'MAPE')[['Product', 'MAPE', 'Accuracy', 'Peak_Timing_Error_Days']]
        print(worst.to_string(index=False))

        return {
            'avg_mape': results_df['MAPE'].mean(),
            'avg_accuracy': results_df['Accuracy'].mean(),
            'early_detection_rate': results_df['Early_Detection_Success'].mean(),
            'target_met': target_met
        }


# Test validator
if __name__ == "__main__":
    from data_loader import KaggleDataLoader
    from forecasting_model import HybridForecastingModel

    print("Testing Model Validator...")

    # Load data
    loader = KaggleDataLoader()
    df = loader.load_and_merge_all()

    # Initialize model
    model = HybridForecastingModel()

    # Run validation
    validator = ModelValidator()
    results = validator.validate_all_products(df, model, max_products=5)

    if results is not None:
        # Print summary
        summary = validator.print_summary(results)