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
        df = df.sort_values('date').copy()

        # 1. Growth Velocity (40% weight)
        df['mentions_7d_avg'] = df['mentions'].rolling(7, min_periods=1).mean()
        df['growth_rate'] = df['mentions_7d_avg'].pct_change(7).fillna(0) * 100

        # Cap at 300% growth for scoring
        growth_score = np.clip(
            df['growth_rate'] / 300 * self.weights['growth_velocity_weight'] * 100,
            0,
            self.weights['growth_velocity_weight'] * 100
        )

        # 2. Sentiment Polarity (20% weight)
        sentiment_score = df['sentiment'] * self.weights['sentiment_weight'] * 100

        # 3. Saturation Index (20% weight)
        # Lower saturation = higher score (more room to grow)
        df['mentions_max'] = df['mentions'].expanding().max()
        saturation = 1 - (df['mentions'] / (df['mentions_max'] + 1))
        saturation_score = saturation * self.weights['saturation_weight'] * 100

        # 4. Profit Margin Proxy (20% weight)
        # Based on growth acceleration
        df['acceleration'] = df['growth_rate'].diff().fillna(0)
        profit_score = np.clip(
            df['acceleration'] / 50 * self.weights['profit_weight'] * 100,
            0,
            self.weights['profit_weight'] * 100
        )

        # Combined Trend Score (0-100)
        df['trend_score'] = growth_score + sentiment_score + saturation_score + profit_score
        df['trend_score'] = np.clip(df['trend_score'], 0, 100)

        # Add components for analysis
        df['growth_component'] = growth_score
        df['sentiment_component'] = sentiment_score
        df['saturation_component'] = saturation_score
        df['profit_component'] = profit_score

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