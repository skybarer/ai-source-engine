"""
Data Loader: Load and preprocess Kaggle datasets
Converts raw e-commerce data into trend time series
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from config import KAGGLE_DATA, PROCESSED_DATA_DIR, PRODUCT_CATEGORIES


class KaggleDataLoader:
    """Load and preprocess e-commerce data from Kaggle"""

    def __init__(self):
        self.processed_data = None

    def load_amazon_data(self):
        """
        Load Amazon India dataset from Kaggle
        Expected columns: product_name, rating, review_date, review_text
        """
        print("Loading Amazon data...")
        try:
            df = pd.read_csv(KAGGLE_DATA['amazon'])

            # Common column name variations
            date_cols = ['review_date', 'date', 'timestamp', 'Date']
            product_cols = ['product_name', 'product', 'Product Name', 'title']
            rating_cols = ['rating', 'Rating', 'stars']

            # Find correct column names
            date_col = next((col for col in date_cols if col in df.columns), None)
            product_col = next((col for col in product_cols if col in df.columns), None)
            rating_col = next((col for col in rating_cols if col in df.columns), None)

            if not all([product_col, rating_col]):
                # Use available data with synthetic dates if needed
                print(f"[OK] Using available Amazon columns with synthetic timeline")
                if not date_col:
                    end_date = datetime.now()
                    df['date'] = [end_date - timedelta(days=np.random.randint(0, 180))
                                  for _ in range(len(df))]
                    date_col = 'date'
            elif not date_col:
                print(f"[OK] Generating synthetic timeline for Amazon data")
                end_date = datetime.now()
                df['date'] = [end_date - timedelta(days=np.random.randint(0, 180))
                              for _ in range(len(df))]
                date_col = 'date'

            # Standardize columns
            df = df.rename(columns={
                date_col: 'date',
                product_col: 'product',
                rating_col: 'rating'
            })

            # Convert date
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])

            # Convert rating to numeric and calculate sentiment (0-1 scale)
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['sentiment'] = df['rating'] / 5.0
            df = df.dropna(subset=['rating', 'sentiment'])

            # Aggregate by date and product
            trend_df = df.groupby(['date', 'product']).agg({
                'rating': 'count',  # mentions
                'sentiment': 'mean'
            }).reset_index()

            trend_df = trend_df.rename(columns={'rating': 'mentions'})
            trend_df['source'] = 'amazon'

            print(f"[OK] Loaded {len(trend_df)} Amazon trend records")
            return trend_df

        except FileNotFoundError:
            print(f"[FAIL] Amazon data not found at {KAGGLE_DATA['amazon']}")
            print("   Download from: https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset")
            return None
        except Exception as e:
            print(f"[FAIL] Error loading Amazon data: {e}")
            return None

    def load_flipkart_data(self):
        """Load Flipkart dataset from Kaggle"""
        print("Loading Flipkart data...")
        try:
            df = pd.read_csv(KAGGLE_DATA['flipkart'])

            # Flipkart data might not have dates - generate synthetic timeline
            if 'date' not in df.columns:
                print("[OK] Generating synthetic timeline for Flipkart data")
                # Assign random dates over past 180 days
                end_date = datetime.now()
                df['date'] = [end_date - timedelta(days=np.random.randint(0, 180))
                              for _ in range(len(df))]

            # Find product and rating columns
            product_col = next((col for col in ['product_name', 'product', 'title']
                                if col in df.columns), None)
            rating_col = next((col for col in ['rating', 'product_rating']
                               if col in df.columns), None)

            if not product_col:
                print(f"[FAIL] Could not find product column in Flipkart data")
                return None

            df = df.rename(columns={product_col: 'product'})

            if rating_col:
                df['sentiment'] = pd.to_numeric(df[rating_col], errors='coerce') / 5.0
            else:
                df['sentiment'] = np.random.uniform(0.6, 0.9, len(df))  # Synthetic

            df['date'] = pd.to_datetime(df['date'])

            # Aggregate
            trend_df = df.groupby(['date', 'product']).size().reset_index(name='mentions')
            trend_df['sentiment'] = df.groupby(['date', 'product'])['sentiment'].mean().values
            trend_df['source'] = 'flipkart'

            print(f"[OK] Loaded {len(trend_df)} Flipkart trend records")
            return trend_df

        except FileNotFoundError:
            print(f"[WARN]  Flipkart data not found (optional)")
            return None
        except Exception as e:
            print(f"[WARN]  Error loading Flipkart data: {e}")
            return None

    def generate_synthetic_data(self, n_products=20, days=180):
        """
        Generate synthetic viral product data
        Use this if Kaggle data is not available
        """
        print(f"Generating synthetic data for {n_products} products...")
        np.random.seed(42)
        data_list = []

        products = [f'Product_{i:02d}' for i in range(n_products)]

        for product in products:
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

            # Viral pattern parameters
            trend_start = np.random.randint(40, 70)
            peak_day = trend_start + np.random.randint(30, 50)

            mentions = np.zeros(days)
            sentiment = np.zeros(days)

            for i in range(days):
                if i < trend_start:
                    # Pre-viral phase
                    mentions[i] = np.random.poisson(5)
                    sentiment[i] = np.random.uniform(0.5, 0.7)
                elif i < peak_day:
                    # Growth phase
                    growth = np.exp((i - trend_start) / 15)
                    mentions[i] = 5 * growth + np.random.normal(0, growth * 0.2)
                    sentiment[i] = np.random.uniform(0.7, 0.95)
                else:
                    # Post-peak saturation
                    decay = np.exp(-(i - peak_day) / 25)
                    mentions[i] = mentions[peak_day - 1] * decay + np.random.normal(0, 10)
                    sentiment[i] = np.random.uniform(0.6, 0.8)

            # Add weekly seasonality
            seasonality = 15 * np.sin(2 * np.pi * np.arange(days) / 7)
            mentions = np.maximum(mentions + seasonality, 0)

            df = pd.DataFrame({
                'date': dates,
                'product': product,
                'mentions': mentions,
                'sentiment': sentiment,
                'peak_day': peak_day,
                'source': 'synthetic'
            })
            data_list.append(df)

        result = pd.concat(data_list, ignore_index=True)
        print(f"[OK] Generated {len(result)} synthetic trend records")
        return result

    def load_and_merge_all(self):
        """
        Load all available datasets and merge
        Falls back to synthetic data if Kaggle data unavailable
        """
        print("=" * 60)
        print("LOADING DATA FROM MULTIPLE SOURCES")
        print("=" * 60)

        dfs = []

        # Try loading Kaggle datasets
        amazon_df = self.load_amazon_data()
        if amazon_df is not None:
            dfs.append(amazon_df)

        flipkart_df = self.load_flipkart_data()
        if flipkart_df is not None:
            dfs.append(flipkart_df)

        # If no real data, use synthetic
        if len(dfs) == 0:
            print("\n[WARN]  No Kaggle data found. Using synthetic data for demonstration.")
            print("   To use real data, download Kaggle datasets to data/raw/\n")
            synthetic_df = self.generate_synthetic_data(n_products=20, days=180)
            dfs.append(synthetic_df)

        # Merge all sources
        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df.sort_values(['product', 'date']).reset_index(drop=True)

        # Save processed data
        output_path = PROCESSED_DATA_DIR / "trend_data.csv"
        merged_df.to_csv(output_path, index=False)
        print(f"\n[OK] Processed data saved to {output_path}")
        print(f"[OK] Total products: {merged_df['product'].nunique()}")
        print(f"[OK] Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")

        self.processed_data = merged_df
        return merged_df

    def get_product_list(self):
        """Get list of products for analysis"""
        if self.processed_data is None:
            self.load_and_merge_all()

        return self.processed_data['product'].unique()

    def get_product_data(self, product_name):
        """Get time series for specific product"""
        if self.processed_data is None:
            self.load_and_merge_all()

        product_df = self.processed_data[
            self.processed_data['product'] == product_name
            ].copy()

        return product_df.sort_values('date').reset_index(drop=True)


# Quick test function
if __name__ == "__main__":
    loader = KaggleDataLoader()
    df = loader.load_and_merge_all()

    print("\n" + "=" * 60)
    print("DATA SAMPLE")
    print("=" * 60)
    print(df.head(10))

    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    print(df.describe())