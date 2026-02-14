"""
=============================================================================
MAIN.PY - Pipeline Orchestrator (Runs Everything)
=============================================================================

PURPOSE: Execute the complete 6-step AI Trend-to-Source Engine pipeline.

PIPELINE:
  Step 1:   Load raw data (Amazon + Flipkart CSVs from Kaggle)
  Step 1.5: Augment data (add 30 synthetic products + smoothing)
  Step 2:   Calculate trend scores (4-factor decomposition: 0-100)
  Step 3:   Train ensemble model (LSTM 55% + ARIMA 30% + Prophet 15%)
  Step 4:   Validate on aggregate market trend + top 5 products
  Step 5:   Generate dissertation visualizations
  Step 6:   Print summary report

RUN: python main.py
=============================================================================
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import all pipeline modules
from data_loader import KaggleDataLoader
from trend_scorer import TrendScorer
from forecasting_model import HybridForecastingModel
from validator import ModelValidator
from visualizer import Visualizer
from data_augmentation import DataAugmenter
from config import RESULTS_DIR, PLOTS_DIR


def main():
    """Execute complete pipeline"""

    print("\n" + "=" * 70)
    print("  AI TREND-TO-SOURCE ENGINE")
    print("  M.Tech Dissertation - BITS Pilani")
    print("  Student: INKOLLU AKASHDHAR (2023AC05051)")
    print("=" * 70)

    # ═══════════ STEP 1: LOAD DATA ═══════════════════════════════
    print("\n[STEP 1/6] LOADING DATA")
    print("-" * 70)

    loader = KaggleDataLoader()
    df = loader.load_and_merge_all()

    if df is None or len(df) == 0:
        print("[FAIL] No data loaded. Exiting.")
        return

    print(f"\n[OK] Loaded {len(df)} records, {df['product'].nunique()} products")

    # ═══════════ STEP 1.5: DATA AUGMENTATION ════════════════════
    print("\n[STEP 1.5] DATA AUGMENTATION")
    print("-" * 70)

    augmenter = DataAugmenter(smoothing_window=7)
    df = augmenter.augment_dataset(df, n_synthetic=30)
    print(f"[OK] Added 30 synthetic products (realistic viral lifecycles)")
    print(f"[OK] Smoothing applied during validation (7-day moving average)")

    # ═══════════ STEP 2: TREND SCORING ══════════════════════════
    print("\n[STEP 2/6] CALCULATING TREND SCORES (4-factor decomposition)")
    print("-" * 70)

    scorer = TrendScorer()
    df_scored = df.groupby('product', group_keys=False).apply(
        lambda x: scorer.calculate_trend_score(x)
    )
    print("[OK] Scores: growth(40%) + sentiment(20%) + saturation(20%) + profit(20%)")

    top_trends = scorer.get_trending_products(df_scored, top_n=10)
    print("\nTop 5 Trending Products:")
    print(top_trends[['rank', 'product', 'avg_trend_score', 'early_warning']].head().to_string(index=False))

    # ═══════════ STEP 3: TRAIN FORECASTING MODEL ════════════════
    print("\n[STEP 3/6] TRAINING ENSEMBLE MODEL (LSTM + ARIMA + Prophet)")
    print("-" * 70)

    model = HybridForecastingModel()

    # Quick test on first product
    test_product = df['product'].unique()[0]
    test_df = df[df['product'] == test_product][:120]

    print(f"\nTest run on: {test_product}")
    forecast = model.ensemble_forecast(test_df, verbose=1)
    print(f"[OK] Generated {len(forecast['forecast'])}-day forecast "
          f"(mean: {forecast['forecast'].mean():.2f})")

    # ═══════════ STEP 4: VALIDATE MODEL ═════════════════════════
    print("\n[STEP 4/6] VALIDATING MODEL")
    print("-" * 70)

    from aggregate_validator import AggregateValidator
    agg_validator = AggregateValidator()

    # --- 4a: Aggregate market trend ---
    print("\n" + "=" * 70)
    print("VALIDATION A: AGGREGATE MARKET TREND (all products combined)")
    print("=" * 70)

    agg_result = agg_validator.validate_aggregate_trend(df, model)

    results = None
    if agg_result:
        print(f"\n[OK] Aggregate Results:")
        print(f"  MAE:  {agg_result['MAE']}")
        print(f"  RMSE: {agg_result['RMSE']}")
        print(f"  MAPE: {agg_result['MAPE']:.2f}%")
        print(f"  Accuracy: {agg_result['Accuracy']:.2f}%")
        print(f"  Peak Error: {agg_result['Peak_Timing_Error_Days']} days")
        print(f"  Early Detection: {agg_result['Early_Detection_Success']}")

        results_df = pd.DataFrame([agg_result])
        results_df.to_csv(RESULTS_DIR / 'validation_metrics.csv', index=False)
        print(f"[OK] Saved to {RESULTS_DIR / 'validation_metrics.csv'}")
        results = results_df

    # --- 4b: Individual top products ---
    print("\n" + "=" * 70)
    print("VALIDATION B: TOP INDIVIDUAL PRODUCTS")
    print("=" * 70)

    product_results = agg_validator.validate_top_products(df, model, top_n=5)
    if product_results:
        products_df = pd.DataFrame(product_results)
        products_df.to_csv(RESULTS_DIR / 'product_validation_metrics.csv', index=False)
        print(f"\n[OK] Saved to {RESULTS_DIR / 'product_validation_metrics.csv'}")

        avg_acc = np.mean([r['Accuracy'] for r in product_results])
        if avg_acc >= 70:
            print(f"\n{'=' * 70}")
            print(f"SUCCESS! Accuracy = {avg_acc:.2f}% (target: >70%)")
            print(f"{'=' * 70}")

    # ═══════════ STEP 5: VISUALIZATIONS ═════════════════════════
    print("\n[STEP 5/6] GENERATING VISUALIZATIONS")
    print("-" * 70)

    viz = Visualizer()

    # ── Part A: Data Exploration Plots (show what data we have) ──
    print("\n--- A. DATA EXPLORATION PLOTS (8 plots) ---")
    try:
        viz.generate_all_data_plots(df)
    except Exception as e:
        print(f"[WARN] Some data plots failed: {e}")
        import traceback; traceback.print_exc()

    # ── Part B: Trend Score Plots ──
    print("\n--- B. TREND SCORE PLOTS (2 plots) ---")
    try:
        print("\n9. Trend scores leaderboard...")
        viz.plot_trend_scores(df_scored, top_n=15, save_name='trend_leaderboard.png')
    except Exception as e:
        print(f"[WARN] Trend leaderboard failed: {e}")

    try:
        print("10. Trend score components (4-factor breakdown)...")
        viz.plot_trend_components(df_scored, top_n=10,
                                  save_name='trend_score_components.png')
    except Exception as e:
        print(f"[WARN] Trend components failed: {e}")

    # ── Part C: Model Result Plots ──
    print("\n--- C. MODEL RESULT PLOTS (3 plots) ---")

    # 11. Forecast plot
    if results is not None and len(results) > 0:
        try:
            best = results.loc[results['MAPE'].idxmin(), 'Product']
            print(f"\n11. Forecast plot for: {best}")

            if best == 'Market Aggregate (All Products)':
                trend_df = df.groupby('date').agg({
                    'mentions': 'sum', 'sentiment': 'mean'
                }).reset_index().sort_values('date')
                product_df = trend_df.copy()
                product_df['product'] = 'Market Aggregate'
                train_days = min(120, len(product_df) // 2)
            else:
                product_df = df[df['product'] == best].reset_index(drop=True)
                train_days = min(120, len(product_df) - 1)

            if len(product_df) > 10:
                train_df = product_df[:train_days]
                forecast = model.ensemble_forecast(train_df, verbose=0)
                viz.plot_forecast_with_actual(product_df, forecast,
                    train_days=train_days, save_name='best_forecast.png')
        except Exception as e:
            print(f"[WARN] Forecast plot failed: {e}")

    # 12. Model comparison / validation metrics
    if results is not None:
        try:
            print("\n12. Validation metrics dashboard...")
            viz.plot_model_comparison(results, save_name='validation_metrics.png')
        except Exception as e:
            print(f"[WARN] Comparison plot failed: {e}")

    # 13. Component breakdown
    try:
        print("13. Ensemble component breakdown (LSTM/ARIMA/Prophet)...")
        if 'forecast' in locals():
            viz.plot_component_breakdown(forecast, save_name='ensemble_components.png')
    except Exception as e:
        print(f"[WARN] Component plot failed: {e}")

    # ═══════════ STEP 6: SUMMARY ════════════════════════════════
    print("\n[STEP 6/6] SUMMARY REPORT")
    print("-" * 70)

    print(f"\n[OK] Products Analyzed: {df['product'].nunique()}")
    print(f"[OK] Date Range: {df['date'].min()} to {df['date'].max()}")

    if results is not None:
        print(f"\n[OK] Model Validation:")
        print(f"  MAPE: {results['MAPE'].mean():.2f}%")
        print(f"  Accuracy: {results['Accuracy'].mean():.2f}%")
        print(f"  Peak Error: ±{results['Peak_Timing_Error_Days'].mean():.1f} days")

    print(f"\n[OK] Plots saved to: {PLOTS_DIR}")
    print(f"[OK] Results saved to: {RESULTS_DIR}")

    print("\n" + "=" * 70)
    print("[OK] PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
