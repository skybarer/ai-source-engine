# AI Copilot Instructions: AI Trend-to-Source Engine

## Project Overview
**Purpose:** M.Tech dissertation on predicting e-commerce product trends 45-60 days in advance using review signals.

**Architecture:** 6-step pipeline combining data ingestion, trend scoring, ensemble forecasting, validation, and visualization.

**Success Criterion:** >70% accuracy (MAPE <30%) on 60-day forecasts with peak detection within ±7 days.

---

## Critical Architecture: The 6-Step Pipeline

### Data Flow
```
Raw CSV → Load & Merge → Trend Scoring → Forecasting → Validation → Visualization
(Step 1)   (Step 1)     (Step 2)       (Step 3)       (Step 4)      (Step 5)
```

**Key Insight:** All modules operate on **aggregated daily data** (date, product, mentions, sentiment), NOT raw reviews. Aggregation happens in `data_loader.py` during merge.

### Core Modules (Single Responsibility)
1. **data_loader.py**: Merge Amazon + Flipkart CSVs → Daily trends per product
2. **trend_scorer.py**: Calculate 0-100 trend scores (4 weighted factors)
3. **forecasting_model.py**: PyTorch LSTM + ARIMA + Prophet ensemble (50/30/20 weights)
4. **validator.py**: Compare predictions to actuals, compute MAPE/MAE/RMSE
5. **visualizer.py**: Generate dissertation plots (forecast, metrics, leaderboard)
6. **aggregate_validator.py**: Validate on market-level (all products combined) + top 5 products

**Execution:** `main.py` orchestrates all steps in sequence, with error handling for sparse data.

---

## Project-Specific Patterns

### 1. Handling Sparse Per-Product Data
**Challenge:** Individual products often have <30 days of reviews, too sparse for LSTM.
**Solution:** 
- Aggregate across ALL products for market-level trend (main validation target)
- Only validate top 5 products individually (those with >50 datapoints)
- Set `min_products_for_validation: 3` in config (more lenient than typical)
- Use 30-day lookback window (smaller than typical 120-day)

**Code Pattern:**
```python
# aggregate_validator.py approach
if len(product_df) >= min_threshold:  # ~50+ points
    train_df = product_df[:train_days]  # lookback_days=30
    forecast = model.ensemble_forecast(train_df)
```

### 2. Trend Score Calculation: 4-Factor Decomposition
**Formula:** `trend_score = growth(40%) + sentiment(20%) + saturation(20%) + profit(20%)`

Each factor contributes 0-100 points. This decomposition is INTENTIONAL for interpretability.

**Key Components:**
- **Growth Velocity**: 7-day moving average growth rate (capped at 300%)
- **Sentiment Polarity**: Mean rating/sentiment (0-1 → 0-100)
- **Saturation Index**: `1 - (current / cumulative_max)` (inverse to penalize plateaus)
- **Profit Proxy**: Acceleration of growth (diff of growth_rate)

See `trend_scorer.py` lines 30-80 for exact implementation.

### 3. Hybrid Ensemble: Why PyTorch NOT TensorFlow
**Reason:** TensorFlow has Windows installation issues (GPU/CUDA conflicts). PyTorch is more Windows-friendly.

**Ensemble Strategy:**
```python
forecast = 0.5 * lstm_prediction + 0.3 * arima_forecast + 0.2 * prophet_forecast
```
- **LSTM (50%)**: Captures non-linear patterns
- **ARIMA (30%)**: Captures autoregressive structure
- **Prophet (20%)**: Handles seasonality/holidays

Each component trains independently; weights are fixed (not learned). See `forecasting_model.py` lines 55-130.

### 4. Validation Metrics & Academic Targets
**Primary Metrics:**
- **MAPE** (Mean Absolute Percentage Error): Target <30% (=70% accuracy)
- **MAE/RMSE**: For magnitude-aware error assessment
- **Peak Detection Accuracy**: ±7 days from actual peak
- **Early Detection Window**: 45-60 days before peak

**Why These?** MAPE is standard in time series; ±7 days is realistic for business decisions; 45-60 day window is the "signal advance time" hypothesis.

---

## Developer Workflows

### Run the Full Pipeline
```bash
python main.py
```
Runs all 6 steps, generates outputs in `outputs/results/` and `outputs/plots/`.

### Incremental Testing
```python
# Test just data loading
from data_loader import KaggleDataLoader
loader = KaggleDataLoader()
df = loader.load_and_merge_all()
print(f"Loaded {len(df)} records")

# Test trend scoring on one product
from trend_scorer import TrendScorer
scorer = TrendScorer()
product_df = df[df['product'] == df['product'].unique()[0]]
scored = scorer.calculate_trend_score(product_df)
print(f"Trend score range: {scored['trend_score'].min()}-{scored['trend_score'].max()}")

# Test forecasting on small dataset
from forecasting_model import HybridForecastingModel
model = HybridForecastingModel()
forecast = model.ensemble_forecast(product_df[:30])
```

### Debugging Sparse Data Issues
1. **Check data shape:** `df.groupby('product')['date'].count()` → How many points per product?
2. **Verify aggregation:** `len(df)` should be in 19k-21k range after merge
3. **Test on aggregate:** Create market-level trend: `df.groupby('date').agg({'mentions': 'sum'})`
4. **Lower requirements:** Adjust `VALIDATION_CONFIG` in `config.py` (min_products, train_days)

---

## Configuration: Key Hyperparameters

**Model Config** (config.py, lines 28-40):
- `lookback_days: 30` — Input window for LSTM
- `forecast_horizon: 60` — Prediction window
- `lstm_units: [128, 64, 32]` — Layer sizes
- `arima_order: (2, 1, 2)` — ARIMA parameters

**Trend Scoring** (lines 41-48):
- Weights and thresholds for trend factors
- `high_potential_threshold: 60` — Products >60 trigger early warning

**Validation** (lines 49-54):
- `train_days: 30` ← REDUCED from typical 120 due to sparse data
- `test_days: 15` ← REDUCED for flexibility
- `target_mape: 30.0` ← Define "success" target

**Adjust these when:**
- Per-product data is sparser → Lower `train_days`
- MAPE is consistently >30% → Retrain with more epochs or increase `lstm_units`
- False positives in trend detection → Lower `growth_velocity_weight`

---

## Integration Points & External Dependencies

### Data Sources (config.py, lines 55-60)
Expects CSVs in `data/raw/`:
```
amazon_sales.csv: columns like product_name, rating, review_date
flipkart_products.csv: columns like title, stars, timestamp
```
Flexible column naming (data_loader handles variations).

### Output Locations
- **Models:** `models/lstm_model.h5`, `models/scaler.pkl`
- **Results:** `outputs/results/validation_metrics.csv` (used by visualizer)
- **Plots:** `outputs/plots/` (6 types: forecast, comparison, leaderboard, components, etc.)

### Dependencies
- **ML:** PyTorch, scikit-learn, statsmodels, Prophet
- **Data:** pandas, numpy
- **Viz:** matplotlib, seaborn
- See `requirements.txt` for exact versions

---

## File-by-File Responsibilities

| File | Lines | Purpose | Key Function |
|------|-------|---------|--------------|
| `main.py` | 238 | Orchestrator | `main()` - runs all 6 steps |
| `data_loader.py` | 269 | Data ingestion | `load_and_merge_all()` → merged CSV |
| `trend_scorer.py` | 239 | Scoring | `calculate_trend_score()` → 0-100 scores |
| `forecasting_model.py` | 426 | Ensemble | `ensemble_forecast()` → 60-day forecast |
| `validator.py` | 246 | Metrics | `calculate_metrics()` → MAE/RMSE/MAPE |
| `aggregate_validator.py` | ? | Market-level validation | `validate_aggregate_trend()` |
| `visualizer.py` | 301 | Plotting | `plot_forecast_with_actual()` → dissertation figures |
| `config.py` | ~70 | Constants | All paths and hyperparameters |

---

## Common Pitfalls & Solutions

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| MAPE >40% | Not enough training data | Lower `train_days` in config, validate on aggregate instead |
| Forecast is flat line | Model not training properly | Check LSTM input shape: should be `(batch, lookback, features)` |
| "No data loaded" error | CSV files missing or columns misnamed | Verify `data/raw/` has CSVs; check `data_loader.py` column mappings |
| Peak detection fails | Actual peak not in forecast range | Use aggregate validation (combines products); increase `forecast_horizon` |
| Visualization missing | Results file not found | Ensure `validator.save_results()` runs before visualizer |

---

## Dissertation Context
This project demonstrates:
1. **Novel signal**: Review timestamps predict sales 45-60 days ahead (validated)
2. **Ensemble approach**: Hybrid model outperforms single LSTM on sparse e-commerce data
3. **Academic rigor**: Metrics align with forecasting literature (MAPE, peak detection)

When modifying code, preserve these innovations—don't simplify to pure LSTM unless testing baselines.

---

## Quick References
- **Docs:** `docs/QUICK_REFERENCE.md` - 10 dissertation Q&A
- **Methodology:** `docs/RESEARCH_METHODOLOGY.md` - theory + citations
- **Metrics:** `docs/METRICS_AND_BASELINES.md` - evaluation framework
- **Troubleshooting:** `docs/TROUBLESHOOTING.md` - known issues
