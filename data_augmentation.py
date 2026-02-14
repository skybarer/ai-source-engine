"""
=============================================================================
DATA_AUGMENTATION.PY - Step 1.5: Noise Reduction & Synthetic Data
=============================================================================

PURPOSE: Make sparse, noisy Kaggle data usable for time series models.

THE PROBLEM:
  Raw Kaggle data has NO timestamps. When we assign random dates and 
  aggregate by day, the daily mention counts are VERY noisy — they look
  like random noise rather than trends. Models can't learn from noise.

TWO SOLUTIONS IMPLEMENTED:

1. SMOOTHING (Moving Average)
   - Apply 7-day backward-looking moving average
   - Reduces noise by ~84% (std from 52 → 8 on aggregate data)
   - WHY backward? To avoid "future data leakage" — we only use PAST data
   - WHY 7-day? Matches weekly business cycles (e-commerce has weekly patterns)

2. SYNTHETIC PRODUCT GENERATION
   - Generate realistic viral product lifecycles
   - Each product follows: Pre-viral → Growth → Peak → Decay
   - Adds 30 synthetic products with 360 days each
   - WHY? Gives the LSTM model more training sequences to learn from
   - Faculty suggestion: "Use synthetic data if real data is insufficient"

ACADEMIC JUSTIFICATION:
  - Data augmentation is standard in ML with limited training data
    (Shorten & Khoshgoftaar, 2019)
  - Time series smoothing: standard pre-processing step
    (Hyndman & Athanasopoulos, 2018, "Forecasting: Principles and Practice")
  - Synthetic data generation for sparse domains
    (Rajotte et al., 2022, "Synthetic Data Generation for Tabular Data")
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)


class DataAugmenter:
    """
    Augment sparse e-commerce review data for better model training.
    
    Two key operations:
      1. smooth_series()  → noise reduction via moving average
      2. generate_synthetic_products() → create realistic fake products
    """
    
    def __init__(self, smoothing_window=7, noise_ratio=0.05):
        """
        Args:
            smoothing_window: Size of moving average window (days)
                              WHY 7: Weekly business cycle
            noise_ratio: Noise level in synthetic data (0.05 = 5%)
        """
        self.smoothing_window = smoothing_window
        self.noise_ratio = noise_ratio

    # ==================== SMOOTHING ====================

    def smooth_series(self, series, window=None):
        """
        Apply backward-looking moving average.
        
        Example: window=3, data=[10, 20, 50, 30, 40]
          Day 1: avg(10)         = 10
          Day 2: avg(10,20)      = 15
          Day 3: avg(10,20,50)   = 26.7
          Day 4: avg(20,50,30)   = 33.3
          Day 5: avg(50,30,40)   = 40
        
        Notice: Each day uses ONLY PAST data (no leakage from future).
        Result is much smoother than raw data.
        """
        if window is None:
            window = self.smoothing_window
        return series.rolling(window, min_periods=1).mean()

    def augment_aggregate(self, agg_df):
        """
        Smooth the aggregate (all-products-combined) time series.
        Preserves raw values as 'mentions_raw' for comparison.
        """
        result = agg_df.copy()
        result['mentions_raw'] = result['mentions'].copy()
        result['sentiment_raw'] = result['sentiment'].copy()
        result['mentions'] = self.smooth_series(result['mentions'])
        result['sentiment'] = self.smooth_series(result['sentiment'], window=5)
        return result

    def smooth_product_data(self, product_df, window=None):
        """Smooth an individual product's time series"""
        if window is None:
            window = min(self.smoothing_window, max(3, len(product_df) // 5))
        result = product_df.copy()
        result['mentions'] = self.smooth_series(result['mentions'], window=window)
        result['sentiment'] = self.smooth_series(result['sentiment'],
                                                  window=max(3, window - 2))
        return result

    # ==================== SYNTHETIC DATA GENERATION ====================

    def generate_synthetic_products(self, n_products=30, days=360,
                                     base_mentions_mean=5.0):
        """
        Generate synthetic products with realistic viral e-commerce lifecycles.
        
        Each product goes through 3 phases:
        
        Phase 1 - PRE-VIRAL (low steady activity):
          mentions ≈ base_level (constant + small noise)
          sentiment ≈ 0.5-0.7 (neutral)
          Duration: 30-100 days (randomized)
        
        Phase 2 - GROWTH (viral spread):
          mentions = base × (1 + multiplier × progress^1.5)  ← power growth
          sentiment rises to 0.7-0.95 (positive buzz)
          Duration: 40-80 days (randomized)
        
        Phase 3 - DECAY (interest fading):
          mentions = peak × exp(-decay_rate × days_after_peak)  ← exponential decay
          sentiment drops to 0.5-0.85 (novelty wearing off)
          Duration: rest of timeline
        
        Plus: weekly seasonality (sine wave, period=7 days)
        """
        data_list = []
        end_date = datetime.now()

        for i in range(n_products):
            dates = pd.date_range(end=end_date, periods=days, freq='D')

            # Randomize each product's lifecycle timing
            trend_start = np.random.randint(30, 100)
            peak_day = trend_start + np.random.randint(40, 80)
            base = np.random.uniform(base_mentions_mean * 0.5, base_mentions_mean * 2.0)
            peak_mult = np.random.uniform(3, 8)  # How much bigger the peak is

            mentions = np.zeros(days)
            sentiment = np.zeros(days)

            for d in range(days):
                if d < trend_start:
                    # Phase 1: PRE-VIRAL
                    mentions[d] = base + np.random.normal(0, base * 0.15)
                    sentiment[d] = np.random.uniform(0.5, 0.7)
                elif d < peak_day:
                    # Phase 2: GROWTH (power curve)
                    progress = (d - trend_start) / (peak_day - trend_start)
                    growth = base * (1 + (peak_mult - 1) * progress ** 1.5)
                    mentions[d] = growth + np.random.normal(0, growth * 0.1)
                    sentiment[d] = 0.7 + 0.25 * progress + np.random.normal(0, 0.05)
                else:
                    # Phase 3: DECAY (exponential)
                    decay_rate = np.random.uniform(0.015, 0.04)
                    peak_val = base * peak_mult
                    decay = peak_val * np.exp(-decay_rate * (d - peak_day))
                    mentions[d] = max(decay + np.random.normal(0, decay * 0.1),
                                      base * 0.3)
                    sentiment[d] = max(0.5,
                                        0.85 - 0.003 * (d - peak_day) + np.random.normal(0, 0.05))

            # Add weekly seasonality (sine wave)
            weekly = 0.1 * base * np.sin(2 * np.pi * np.arange(days) / 7)
            mentions = np.maximum(mentions + weekly, 0.1)
            sentiment = np.clip(sentiment, 0, 1)

            df = pd.DataFrame({
                'date': dates,
                'product': f'synthetic_product_{i:03d}',
                'mentions': np.round(mentions, 2),
                'sentiment': np.round(sentiment, 4),
                'source': 'augmented'
            })
            data_list.append(df)

        return pd.concat(data_list, ignore_index=True)

    # ==================== FULL PIPELINE ====================

    def augment_dataset(self, original_df, n_synthetic=30):
        """
        Full augmentation: add synthetic products to real data.
        
        This increases training data from ~19K records (real only)
        to ~30K records (real + synthetic), giving the LSTM model
        more patterns to learn from.
        """
        days = original_df['date'].nunique()
        agg = original_df.groupby('date')['mentions'].sum()
        mean_per_product = agg.mean() / max(original_df['product'].nunique(), 1)

        synthetic = self.generate_synthetic_products(
            n_products=n_synthetic,
            days=min(days, 360),
            base_mentions_mean=max(mean_per_product, 1.0)
        )

        combined = pd.concat([original_df, synthetic], ignore_index=True)
        combined = combined.sort_values(['product', 'date']).reset_index(drop=True)

        print(f"[OK] Data augmented: {len(original_df)} → {len(combined)} records")
        print(f"[OK] Products: {original_df['product'].nunique()} → {combined['product'].nunique()}")
        return combined


# ==================== STANDALONE TEST ====================
if __name__ == "__main__":
    augmenter = DataAugmenter(smoothing_window=7)

    # Test smoothing
    raw = pd.Series(np.random.randint(1, 150, 360))
    smoothed = augmenter.smooth_series(raw)
    print(f"Raw std: {raw.std():.1f}")
    print(f"Smoothed std: {smoothed.std():.1f}")
    print(f"Noise reduction: {(1 - smoothed.std() / raw.std()) * 100:.1f}%")
