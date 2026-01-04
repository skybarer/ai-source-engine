#!/usr/bin/env python
"""
Simplified Main Execution Script - No Blocking Operations
Run complete AI Trend-to-Source Engine pipeline with minimal dependencies
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Disable matplotlib interactive mode to prevent blocking
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from data_loader import KaggleDataLoader
from trend_scorer import TrendScorer
from forecasting_model import HybridForecastingModel
from aggregate_validator import AggregateValidator
from visualizer import Visualizer
from config import RESULTS_DIR, PLOTS_DIR


def main():
    """Execute complete pipeline without blocking"""

    print("\n" + "=" * 70)
    print("  AI TREND-TO-SOURCE ENGINE")
    print("  M.Tech Dissertation - BITS Pilani")
    print("  Student: INKOLLU AKASHDHAR (2023AC05051)")
    print("=" * 70)
    sys.stdout.flush()

    # ========== STEP 1: LOAD DATA ==========
    print("\n[STEP 1/6] LOADING DATA")
    print("-" * 70)
    sys.stdout.flush()

    try:
        loader = KaggleDataLoader()
        df = loader.load_and_merge_all()

        if df is None or len(df) == 0:
            print("[FAIL] No data loaded. Exiting.")
            return

        print(f"\n[OK] Loaded {len(df)} records across {df['product'].nunique()} products")
        sys.stdout.flush()
    except Exception as e:
        print(f"[FAIL] Data loading failed: {e}")
        return

    # ========== STEP 2: CALCULATE TREND SCORES ==========
    print("\n[STEP 2/6] CALCULATING TREND SCORES")
    print("-" * 70)
    sys.stdout.flush()

    try:
        scorer = TrendScorer()
        df_scored = df.groupby('product', group_keys=False).apply(
            lambda x: scorer.calculate_trend_score(x)
        )
        print("[OK] Trend scores calculated")

        # Get top trending products
        top_trends = scorer.get_trending_products(df_scored, top_n=10)
        print("\nTop 5 Trending Products:")
        print(top_trends[['rank', 'product', 'avg_trend_score', 'early_warning']].head().to_string(index=False))
        sys.stdout.flush()
    except Exception as e:
        print(f"[FAIL] Trend scoring failed: {e}")
        df_scored = df.copy()

    # ========== STEP 3: TRAIN FORECASTING MODEL ==========
    print("\n[STEP 3/6] TRAINING FORECASTING MODELS")
    print("-" * 70)
    sys.stdout.flush()

    try:
        model = HybridForecastingModel()

        # Quick test on one product with >30 points
        products_by_count = df.groupby('product').size().sort_values(ascending=False)
        test_product = products_by_count[products_by_count >= 30].index[0] if len(products_by_count[products_by_count >= 30]) > 0 else df['product'].unique()[0]
        test_df = df[df['product'] == test_product][:120]

        print(f"Testing ensemble on: {test_product[:50]}")
        print(f"Data points available: {len(test_df)}")
        sys.stdout.flush()

        if len(test_df) >= 30:
            forecast = model.ensemble_forecast(test_df, verbose=0)
            print(f"[OK] Generated 60-day forecast (mean: {forecast['forecast'].mean():.2f})")
        else:
            print("[WARN] Not enough data for sample forecast")
            forecast = None
        sys.stdout.flush()
    except Exception as e:
        print(f"[FAIL] Model training failed: {e}")
        model = HybridForecastingModel()
        forecast = None

    # ========== STEP 4: VALIDATE MODEL ==========
    print("\n[STEP 4/6] VALIDATING MODEL PERFORMANCE")
    print("-" * 70)
    sys.stdout.flush()

    try:
        agg_validator = AggregateValidator()
        
        print("\nVALIDATING ON AGGREGATE MARKET TREND (All Products Combined)")
        sys.stdout.flush()
        
        agg_result = agg_validator.validate_aggregate_trend(df, model)
        
        results = None
        if agg_result:
            print(f"\n[OK] Aggregate Validation Results:")
            print(f"  MAE: {agg_result['MAE']:.2f}")
            print(f"  RMSE: {agg_result['RMSE']:.2f}")
            print(f"  MAPE: {agg_result['MAPE']:.2f}%")
            print(f"  Accuracy: {agg_result['Accuracy']:.2f}%")
            print(f"  Peak Timing Error: {agg_result['Peak_Timing_Error_Days']:.1f} days")
            print(f"  Early Detection Success: {agg_result['Early_Detection_Success']}")
            
            # Save results
            results_df = pd.DataFrame([agg_result])
            results_df.to_csv(RESULTS_DIR / 'validation_metrics.csv', index=False)
            print(f"[OK] Results saved to {RESULTS_DIR / 'validation_metrics.csv'}")
            results = results_df
        else:
            print("[FAIL] Aggregate validation failed")
        
        sys.stdout.flush()
        
        # Validate on top products
        print("\nVALIDATING TOP PRODUCTS (With Sufficient Individual Data)")
        sys.stdout.flush()
        
        product_results = agg_validator.validate_top_products(df, model, top_n=5)
        
        if product_results:
            products_df = pd.DataFrame(product_results)
            products_df.to_csv(RESULTS_DIR / 'product_validation_metrics.csv', index=False)
            print(f"[OK] Product results saved to {RESULTS_DIR / 'product_validation_metrics.csv'}")
            
            if len(product_results) > 0:
                avg_accuracy = np.mean([r['Accuracy'] for r in product_results])
                print(f"\n[OK] Average Product Accuracy: {avg_accuracy:.2f}%")
        
        sys.stdout.flush()

    except Exception as e:
        print(f"[FAIL] Validation failed: {e}")
        results = None

    # ========== STEP 5: CREATE VISUALIZATIONS ==========
    print("\n[STEP 5/6] GENERATING VISUALIZATIONS")
    print("-" * 70)
    sys.stdout.flush()

    try:
        viz = Visualizer()

        # 1. Trend scores leaderboard
        try:
            print("1. Creating trend scores leaderboard...")
            viz.plot_trend_scores(df_scored, top_n=15, save_name='trend_leaderboard.png')
            plt.close('all')
            print("[OK] Trend leaderboard created")
        except Exception as e:
            print(f"[WARN] Trend leaderboard failed: {e}")
            plt.close('all')
        sys.stdout.flush()

        # 2. Model comparison
        if results is not None:
            try:
                print("2. Creating model comparison plot...")
                viz.plot_model_comparison(results, save_name='validation_metrics.png')
                plt.close('all')
                print("[OK] Model comparison created")
            except Exception as e:
                print(f"[WARN] Model comparison failed: {e}")
                plt.close('all')
        else:
            print("2. [SKIP] Model comparison (no results)")
        sys.stdout.flush()

        # 3. Data quality
        try:
            print("3. Creating data quality summary...")
            quality_df = pd.DataFrame({
                'Metric': ['Total Records', 'Unique Products', 'Date Range', 'Avg Records/Product'],
                'Value': [
                    len(df),
                    df['product'].nunique(),
                    f"{df['date'].min().date()} to {df['date'].max().date()}",
                    f"{len(df) / df['product'].nunique():.1f}"
                ]
            })
            print(quality_df.to_string(index=False))
            print("[OK] Data quality metrics displayed")
        except Exception as e:
            print(f"[WARN] Data quality summary failed: {e}")
        sys.stdout.flush()

    except Exception as e:
        print(f"[FAIL] Visualization step failed: {e}")

    # ========== STEP 6: SUMMARY ==========
    print("\n[STEP 6/6] GENERATING SUMMARY REPORT")
    print("-" * 70)
    sys.stdout.flush()

    print("\n[SUMMARY] FINAL RESULTS")
    print("=" * 70)
    print(f"[OK] Data Source: {df['source'].unique()[0] if 'source' in df.columns else 'Kaggle/Synthetic'}")
    print(f"[OK] Products Analyzed: {df['product'].nunique()}")
    print(f"[OK] Date Range: {df['date'].min()} to {df['date'].max()}")

    if results is not None and len(results) > 0:
        print(f"\n[OK] Model Validation:")
        print(f"  - Average MAPE: {results['MAPE'].mean():.2f}%")
        print(f"  - Average Accuracy: {results['Accuracy'].mean():.2f}%")
        if 'Early_Detection_Success' in results.columns:
            print(f"  - Early Detection Success: {results['Early_Detection_Success'].mean() * 100:.1f}%")
        if 'Peak_Timing_Error_Days' in results.columns:
            print(f"  - Peak Timing Error: ±{results['Peak_Timing_Error_Days'].mean():.1f} days")

    print(f"\n[OK] Top Trending Products:")
    for i, row in top_trends.head(3).iterrows():
        marker = "[HOT]" if row['early_warning'] else "[TREND]"
        print(f"  {marker} #{row['rank']}: {row['product'][:40]} (Score: {row['avg_trend_score']:.1f})")

    print(f"\n[OK] Output Locations:")
    print(f"  - Visualizations: {PLOTS_DIR}")
    print(f"  - Results CSV: {RESULTS_DIR}")

    print("\n" + "=" * 70)
    print("[SUCCESS] PIPELINE COMPLETE")
    print("=" * 70)

    print("\n[INFO] Next steps:")
    print("  1. Review outputs/plots/ for dissertation figures")
    print("  2. Check outputs/results/validation_metrics.csv for metrics")
    print("  3. All results ready for thesis submission")
    print("\nGood luck with your dissertation!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
