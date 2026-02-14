"""
=============================================================================
AGGREGATE_VALIDATOR.PY - Market-Level + Top Product Validation
=============================================================================

PURPOSE: Validate the model at two levels:
  1. AGGREGATE (all products combined → market-level trend)
  2. INDIVIDUAL (top 5 products with most data)

WHY AGGREGATE VALIDATION?
  Individual products in our Kaggle data are sparse (few days each).
  But when we SUM mentions across ALL products per day, we get a 
  dense, continuous time series representing the overall market trend.
  This is our PRIMARY validation target.

WHY ALSO INDIVIDUAL?
  To show the model works on specific products too, not just aggregate.
  We pick the top 5 products with most records (>10 data points).
=============================================================================
"""

import pandas as pd
import numpy as np
from validator import ModelValidator
from data_augmentation import DataAugmenter


class AggregateValidator(ModelValidator):
    """Extends ModelValidator with aggregate and per-product validation"""
    
    def __init__(self):
        super().__init__()
        self.augmenter = DataAugmenter(smoothing_window=7)
    
    def validate_aggregate_trend(self, data_df, model):
        """
        Validate on aggregate market trend (all products combined).
        
        Process:
          1. Group ALL products by date → sum mentions, mean sentiment
          2. Apply 7-day smoothing (noise reduction)
          3. Split into train/test
          4. Run ensemble forecast on train portion
          5. Compare forecast vs actual test portion
          6. Compute metrics (MAPE, MAE, etc.)
        """
        print("\nAggregating data by date (summing across all products)...")
        
        # Sum mentions across all products for each day
        trend_df = data_df.groupby('date').agg({
            'mentions': 'sum',       # Total mentions per day
            'sentiment': 'mean'      # Average sentiment per day
        }).reset_index().sort_values('date')
        
        print(f"[OK] Aggregate: {len(trend_df)} daily data points")
        
        # Apply smoothing to reduce noise
        trend_df = self.augmenter.augment_aggregate(trend_df)
        print(f"[OK] Applied 7-day moving average smoothing")
        
        if len(trend_df) < 30:
            print(f"[FAIL] Only {len(trend_df)} points (need 30+)")
            return None
        
        # Train/test split
        train_days = min(self.config['train_days'], len(trend_df) // 2)
        test_days = min(self.config['test_days'], len(trend_df) // 3)
        if test_days < 5:
            print(f"[FAIL] Only {test_days} test points (need 5+)")
            return None
        
        train_df = trend_df[:train_days].copy()
        test_df = trend_df[train_days:train_days + test_days].copy()
        actual = test_df['mentions'].values
        
        print(f"  Train: {len(train_df)} days | Test: {len(test_df)} days")
        
        try:
            print("  Generating ensemble forecast...")
            forecast_result = model.ensemble_forecast(train_df, verbose=0)
            predicted = forecast_result['forecast']
            
            metrics = self.calculate_metrics(actual, predicted)
            
            return {
                'Product': 'Market Aggregate (All Products)',
                'Data_Points': len(trend_df),
                'Train_Size': len(train_df),
                'Test_Size': len(test_df),
                'Actual_Mean': round(actual.mean(), 2),
                'Predicted_Mean': round(predicted.mean(), 2),
                **metrics
            }
        except Exception as e:
            print(f"[FAIL] Validation failed: {e}")
            return None
    
    def validate_top_products(self, data_df, model, top_n=5):
        """
        Validate on top N individual products (those with most records).
        
        Prioritizes REAL products over synthetic ones.
        Applies smoothing to each product before validation.
        """
        # Separate real vs synthetic products
        real = data_df[~data_df['product'].str.startswith('synthetic_')]
        synthetic = data_df[data_df['product'].str.startswith('synthetic_')]
        
        # Find products with enough data (min 10 records)
        real_counts = real['product'].value_counts()
        real_top = real_counts[real_counts >= 10].head(top_n)
        
        # Supplement with synthetic if needed
        if len(real_top) < top_n:
            syn_counts = synthetic['product'].value_counts()
            syn_top = syn_counts[syn_counts >= 10].head(top_n - len(real_top))
            top_products = pd.concat([real_top, syn_top])
        else:
            top_products = real_top
        
        if len(top_products) == 0:
            print("[WARN] No products have enough data for individual validation")
            return []
        
        print(f"\n[OK] Found {len(top_products)} products with 10+ records:")
        for i, (name, count) in enumerate(top_products.items(), 1):
            print(f"  {i}. {name[:50]}: {count} records")
        
        results = []
        for idx, (name, count) in enumerate(top_products.items(), 1):
            product_df = data_df[data_df['product'] == name].sort_values('date')
            product_df = self.augmenter.smooth_product_data(product_df)
            
            print(f"\n[{idx}/{len(top_products)}] Validating: {name[:40]}...")
            result = self.validate_single_product(product_df, model, name)
            
            if result:
                results.append(result)
                print(f"  [OK] Accuracy={result['Accuracy']}%, Peak Error={result['Peak_Timing_Error_Days']} days")
            else:
                print(f"  [SKIP] Insufficient processable data")
        
        return results
