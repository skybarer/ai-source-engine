#!/usr/bin/env python
"""Quick test to identify stuck point"""

import sys
from pathlib import Path

print("[1/6] Loading data...", flush=True)
from data_loader import KaggleDataLoader
loader = KaggleDataLoader()
df = loader.load_and_merge_all()
print(f"[OK] Loaded {len(df)} records\n", flush=True)

print("[2/6] Calculating trend scores...", flush=True)
from trend_scorer import TrendScorer
scorer = TrendScorer()
df_scored = df.groupby('product', group_keys=False).apply(
    lambda x: scorer.calculate_trend_score(x)
)
print("[OK] Trend scores calculated\n", flush=True)

print("[3/6] Training forecasting model...", flush=True)
from forecasting_model import HybridForecastingModel
model = HybridForecastingModel()
test_product = df['product'].unique()[0]
test_df = df[df['product'] == test_product][:120]
print(f"Testing on: {test_product[:50]}")
print(f"Data points: {len(test_df)}", flush=True)

if len(test_df) > 30:
    print("Generating ensemble forecast...", flush=True)
    forecast = model.ensemble_forecast(test_df, verbose=1)
    print(f"[OK] Generated 60-day forecast\n", flush=True)
else:
    print("[SKIP] Not enough data\n", flush=True)

print("[4/6] Validating model...", flush=True)
from aggregate_validator import AggregateValidator
agg_validator = AggregateValidator()
agg_result = agg_validator.validate_aggregate_trend(df, model)
if agg_result:
    print(f"[OK] MAPE: {agg_result['MAPE']:.2f}%")
    print(f"[OK] Accuracy: {agg_result['Accuracy']:.2f}%\n", flush=True)
else:
    print("[FAILED] Aggregate validation\n", flush=True)

print("[5/6] Creating visualizations...", flush=True)
from visualizer import Visualizer
viz = Visualizer()
print("[OK] Visualizer initialized\n", flush=True)

print("[6/6] Done!", flush=True)
print("\n[SUCCESS] All steps completed without hanging!")
