"""
Trend Scoring Algorithm
Calculates multi-factor trend scores (0-100) combining:
- Growth Velocity (40%)
- Sentiment Polarity (20%)
- Saturation Index (20%)
- Profit Potential (20%)
"""

import pandas as pd
import numpy as np
from config import TREND_SCORING


class TrendScorer:
    """Calculate trend scores for products"""

    def __init__(self):
        self.weights = TREND_SCORING

    def calculate_trend_score(self, df):
        """
        Calculate comprehensive trend score

        Args:
            df: DataFrame with columns [date, product, mentions, sentiment]

        Returns:
            DataFrame with added trend_score column
        """
        try:
            df = df.sort_values('date').copy().reset_index(drop=True)

            # Ensure numeric types
            df['mentions'] = pd.to_numeric(df['mentions'], errors='coerce').fillna(0)
            df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').fillna(0.5)

            # 1. Growth Velocity (40% weight)
            df['mentions_7d_avg'] = df['mentions'].rolling(7, min_periods=1).mean()
            df['growth_rate'] = df['mentions_7d_avg'].pct_change(7).fillna(0) * 100

            # Cap at 300% growth for scoring
            growth_weight = self.weights['growth_velocity_weight']
            max_growth = growth_weight * 100
            growth_score_vals = df['growth_rate'].values / 300 * max_growth
            growth_score = pd.Series(np.clip(growth_score_vals, 0, max_growth), index=df.index)

            # 2. Sentiment Polarity (20% weight)
            sentiment_clipped = np.clip(df['sentiment'].values, 0, 1)
            sentiment_weight = self.weights['sentiment_weight']
            sentiment_score = pd.Series(sentiment_clipped * sentiment_weight * 100, index=df.index)

            # 3. Saturation Index (20% weight)
            df['mentions_cummax'] = df['mentions'].cummax()
            saturation = 1 - (df['mentions'] / (df['mentions_cummax'] + 1))
            saturation_weight = self.weights['saturation_weight']
            saturation_score = saturation * saturation_weight * 100

            # 4. Profit Margin Proxy (20% weight)
            df['acceleration'] = df['growth_rate'].diff().fillna(0)
            profit_weight = self.weights['profit_weight']
            max_profit = profit_weight * 100
            profit_score_vals = df['acceleration'].values / 50 * max_profit
            profit_score = pd.Series(np.clip(profit_score_vals, 0, max_profit), index=df.index)

            # Combined Trend Score (0-100) - using Series.values for operations
            combined = growth_score.values + sentiment_score.values + saturation_score.values + profit_score.values
            df['trend_score'] = np.clip(combined, 0, 100)

            # Add components for analysis
            df['growth_component'] = np.clip(growth_score.values, 0, None)
            df['sentiment_component'] = np.clip(sentiment_score.values, 0, None)
            df['saturation_component'] = np.clip(saturation_score.values, 0, None)
            df['profit_component'] = np.clip(profit_score.values, 0, None)

            return df
        
        except Exception as e:
            # Fallback: Return dataframe with constant scores
            df['trend_score'] = 50.0
            df['growth_component'] = 0.0
            df['sentiment_component'] = 0.0
            df['saturation_component'] = 0.0
            df['profit_component'] = 0.0
            return df






    def detect_early_warning(self, df, window_days=7):
        """
        Detect if product is in early growth phase
        Returns warning signal for products likely to peak in 45-60 days

        Args:
            df: DataFrame with trend_score column
            window_days: Days to look back for velocity calculation

        Returns:
            dict with warning status and metrics
        """
        if len(df) < window_days * 2:
            return {
                'warning': False,
                'current_score': 0,
                'velocity': 0,
                'reason': 'Insufficient data'
            }

        # Recent trend score
        recent_scores = df['trend_score'].tail(window_days)
        previous_scores = df['trend_score'].tail(window_days * 2).head(window_days)

        current_score = recent_scores.mean()
        previous_score = previous_scores.mean()
        velocity = current_score - previous_score

        # Warning conditions
        high_score = current_score > self.weights['high_potential_threshold']
        accelerating = velocity > self.weights['velocity_threshold']

        warning = high_score and accelerating

        return {
            'warning': warning,
            'current_score': round(current_score, 2),
            'velocity': round(velocity, 2),
            'acceleration': accelerating,
            'reason': 'High potential - early growth detected' if warning else 'Normal'
        }

    def rank_products(self, df_all_products):
        """
        Rank all products by current trend score

        Args:
            df_all_products: DataFrame with multiple products

        Returns:
            DataFrame ranked by trend score
        """
        # Get latest score for each product
        latest_scores = df_all_products.groupby('product').apply(
            lambda x: x.nlargest(7, 'date')['trend_score'].mean()
        ).reset_index(name='avg_trend_score')

        # Sort by score
        ranked = latest_scores.sort_values('avg_trend_score', ascending=False)
        ranked['rank'] = range(1, len(ranked) + 1)

        return ranked

    def get_trending_products(self, df_all_products, top_n=10):
        """
        Get top trending products with early warning signals

        Args:
            df_all_products: DataFrame with all products
            top_n: Number of top products to return

        Returns:
            DataFrame with top trending products and warning status
        """
        results = []

        for product in df_all_products['product'].unique():
            product_df = df_all_products[
                df_all_products['product'] == product
                ].copy()

            if len(product_df) < 14:  # Need at least 2 weeks of data
                continue

            # Calculate trend score if not already done
            if 'trend_score' not in product_df.columns:
                product_df = self.calculate_trend_score(product_df)

            # Get early warning
            warning = self.detect_early_warning(product_df)

            # Latest metrics
            latest = product_df.tail(7)

            results.append({
                'product': product,
                'avg_trend_score': latest['trend_score'].mean(),
                'growth_rate': latest['growth_rate'].mean(),
                'sentiment': latest['sentiment'].mean(),
                'mentions': latest['mentions'].mean(),
                'early_warning': warning['warning'],
                'warning_reason': warning['reason'],
                'score_velocity': warning['velocity']
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('avg_trend_score', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)

        return results_df.head(top_n)


# Test the scorer
if __name__ == "__main__":
    from data_loader import KaggleDataLoader

    print("Testing Trend Scorer...")

    # Load data
    loader = KaggleDataLoader()
    df = loader.load_and_merge_all()

    # Initialize scorer
    scorer = TrendScorer()

    # Calculate scores for all products
    df_scored = df.groupby('product', group_keys=False).apply(
        lambda x: scorer.calculate_trend_score(x)
    )

    # Get top trending
    top_trends = scorer.get_trending_products(df_scored, top_n=5)

    print("\n" + "=" * 60)
    print("TOP 5 TRENDING PRODUCTS")
    print("=" * 60)
    print(top_trends.to_string(index=False))

    # Test early warning for top product
    top_product = top_trends.iloc[0]['product']
    product_data = df_scored[df_scored['product'] == top_product]
    warning = scorer.detect_early_warning(product_data)

    print("\n" + "=" * 60)
    print(f"EARLY WARNING ANALYSIS: {top_product}")
    print("=" * 60)
    for key, value in warning.items():
        print(f"{key}: {value}")