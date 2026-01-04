"""
Aggregate Validation Module
Validates on market-level trends (all products combined)
"""

import pandas as pd
import numpy as np
from validator import ModelValidator


class AggregateValidator(ModelValidator):
    """Validate on aggregate time series data"""
    
    def validate_aggregate_trend(self, data_df, model):
        """
        Validate on aggregate market-level trend
        
        Groups all products by date and sums mentions to create
        a continuous time series for market-level forecasting
        
        Args:
            data_df: Full dataset with all products
            model: Trained forecasting model
            
        Returns:
            dict of validation metrics, or None if failed
        """
        print("\nAggregating data by date...")
        
        # Aggregate by date - sum mentions across all products
        trend_df = data_df.groupby('date').agg({
            'mentions': 'sum',
            'sentiment': 'mean'
        }).reset_index().sort_values('date')
        
        print(f"[OK] Aggregate trend created: {len(trend_df)} daily data points")
        
        # Check if we have enough data
        if len(trend_df) < 30:
            print(f"[FAIL] Insufficient data for validation ({len(trend_df)} points < 30)")
            return None
        
        # Split into train/test using configuration
        train_days = min(self.config['train_days'], len(trend_df) // 2)
        test_days = min(self.config['test_days'], len(trend_df) // 3)
        
        if test_days < 5:
            print(f"[FAIL] Insufficient test data ({test_days} points < 5)")
            return None
        
        train_df = trend_df[:train_days].copy()
        test_df = trend_df[train_days:train_days + test_days].copy()
        
        actual = test_df['mentions'].values
        
        print(f"  - Train size: {len(train_df)} days")
        print(f"  - Test size: {len(test_df)} days")
        
        # Forecast
        try:
            print("  - Generating ensemble forecast...")
            forecast_result = model.ensemble_forecast(train_df, verbose=0)
            predicted = forecast_result['forecast']
            
            # Calculate metrics
            metrics = self.calculate_metrics(actual, predicted)
            
            result = {
                'Product': 'Market Aggregate (All Products)',
                'Data_Points': len(trend_df),
                'Train_Size': len(train_df),
                'Test_Size': len(test_df),
                'Actual_Mean': round(actual.mean(), 2),
                'Predicted_Mean': round(predicted.mean(), 2),
                **metrics
            }
            
            print("  [OK] Validation complete")
            return result
            
        except Exception as e:
            print(f"[FAIL] Validation failed: {e}")
            return None
    
    def validate_top_products(self, data_df, model, top_n=5):
        """
        Validate on top N products (those with most records)
        
        Identifies products with sufficient data and validates
        the ensemble model on each individually
        
        Args:
            data_df: Full dataset with all products
            model: Trained forecasting model
            top_n: Number of top products to validate
            
        Returns:
            list of result dicts (one per product validated)
        """
        # Find products with enough data (minimum 10 records)
        product_counts = data_df['product'].value_counts()
        top_products = product_counts[product_counts >= 10].head(top_n)
        
        if len(top_products) == 0:
            print(f"\n[WARN]  No products have >= 10 records for individual validation")
            print(f"   Max records per product: {product_counts.max()}")
            return []
        
        print(f"\n[OK] Found {len(top_products)} products with >= 10 records:")
        for i, (product_name, count) in enumerate(top_products.items(), 1):
            print(f"  {i}. {product_name}: {count} records")
        
        results = []
        for idx, (product_name, count) in enumerate(top_products.items(), 1):
            product_df = data_df[data_df['product'] == product_name].sort_values('date')
            
            print(f"\n[{idx}/{len(top_products)}] Validating: {product_name[:40]}...")
            
            # Validate
            result = self.validate_single_product(product_df, model, product_name)
            
            if result:
                results.append(result)
                print(f"  [OK] MAE={result['MAE']}, Accuracy={result['Accuracy']}%, "
                      f"Peak Error={result['Peak_Timing_Error_Days']} days")
            else:
                print(f"  ⊘ Validation skipped (insufficient processable data)")
        
        return results
    
    def validate_by_category(self, data_df, model):
        """
        Validate on category-level aggregates
        
        Groups by product category and validates ensemble
        on category-level trends
        
        Args:
            data_df: Full dataset with category info
            model: Trained forecasting model
            
        Returns:
            list of result dicts (one per category)
        """
        if 'category' not in data_df.columns:
            print("[WARN]  No category column in data, skipping category validation")
            return []
        
        categories = data_df['category'].unique()
        print(f"\n[OK] Found {len(categories)} product categories")
        
        results = []
        for category in categories[:5]:  # Top 5 categories
            category_df = data_df[data_df['category'] == category]
            
            # Aggregate by date within category
            trend_df = category_df.groupby('date').agg({
                'mentions': 'sum',
                'sentiment': 'mean'
            }).reset_index().sort_values('date')
            
            if len(trend_df) < 30:
                continue
            
            print(f"\n[Category: {category}] - {len(trend_df)} daily points")
            
            # Split and validate
            train_days = min(self.config['train_days'], len(trend_df) // 2)
            test_days = min(self.config['test_days'], len(trend_df) // 3)
            
            if test_days < 5:
                continue
            
            train_df = trend_df[:train_days]
            test_df = trend_df[train_days:train_days + test_days]
            
            actual = test_df['mentions'].values
            
            try:
                forecast_result = model.ensemble_forecast(train_df, verbose=0)
                predicted = forecast_result['forecast']
                
                metrics = self.calculate_metrics(actual, predicted)
                
                result = {
                    'Product': f'Category: {category}',
                    **metrics,
                    'Data_Points': len(trend_df),
                    'Train_Size': len(train_df),
                    'Test_Size': len(test_df)
                }
                
                results.append(result)
                print(f"  [OK] Accuracy={metrics['Accuracy']}%")
                
            except Exception as e:
                print(f"  [FAIL] Validation failed: {e}")
                continue
        
        return results
