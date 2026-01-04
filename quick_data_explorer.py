"""
Quick Data Explorer: Fast data analysis and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

from config import PROCESSED_DATA_DIR, PLOTS_DIR

def quick_explore():
    """Run quick data exploration"""
    print("\n" + "="*70)
    print("DATA EXPLORATION & QUICK VISUALIZATIONS")
    print("="*70)
    
    # Load data
    try:
        df = pd.read_csv(PROCESSED_DATA_DIR / 'trend_data.csv')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        print(f"\n[OK] Loaded {len(df):,} records")
        print(f"[OK] Total unique products: {df['product'].nunique():,}")
    except FileNotFoundError:
        print(f"[FAIL] Data file not found")
        return
    
    # Print summary
    print("\n" + "-"*70)
    print("DATA SUMMARY")
    print("-"*70)
    
    print(f"\n[CHART] Dataset: {len(df):,} rows × {len(df.columns)} columns")
    print(f"\n📅 Date Range:")
    print(f"   Start: {df['date'].min()}")
    print(f"   End:   {df['date'].max()}")
    print(f"   Duration: {(df['date'].max() - df['date'].min()).days} days")
    
    print(f"\n📦 Products: {df['product'].nunique():,} unique products")
    print(f"   Avg records/product: {len(df) / df['product'].nunique():.2f}")
    
    print(f"\n📈 Mentions:")
    print(f"   Total: {df['mentions'].sum():,}")
    print(f"   Mean: {df['mentions'].mean():.2f}")
    print(f"   Max: {df['mentions'].max():,}")
    
    print(f"\n⭐ Sentiment:")
    print(f"   Mean: {df['sentiment'].mean():.3f}")
    print(f"   Std: {df['sentiment'].std():.3f}")
    
    output_dir = Path(PLOTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Mentions distribution
    print("\n[CHART] Generating visualizations...")
    print("1/5: Mentions distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['mentions'], bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Mentions per Record')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_yscale('log')
    ax.set_title('Distribution of Mentions Across Records')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'data_mentions_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved data_mentions_histogram.png")
    
    # Plot 2: Sentiment distribution
    print("2/5: Sentiment distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['sentiment'], bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax.axvline(df['sentiment'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {df["sentiment"].mean():.3f}')
    ax.set_xlabel('Sentiment Score (0-1)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Sentiment Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'data_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved data_sentiment_distribution.png")
    
    # Plot 3: Top products
    print("3/5: Top products by mentions...")
    fig, ax = plt.subplots(figsize=(12, 6))
    top_products = df.groupby('product')['mentions'].sum().sort_values(ascending=False).head(15)
    ax.barh(range(len(top_products)), top_products.values, color='coral', edgecolor='black')
    ax.set_yticks(range(len(top_products)))
    ax.set_yticklabels([name[:40] for name in top_products.index], fontsize=9)
    ax.set_xlabel('Total Mentions')
    ax.set_title('Top 15 Products by Total Mentions')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'data_top_products.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved data_top_products.png")
    
    # Plot 4: Daily trends
    print("4/5: Daily trends...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    daily_stats = df.groupby('date').agg({
        'mentions': ['sum', 'mean', 'count']
    }).reset_index()
    daily_stats.columns = ['date', 'total_mentions', 'mean_mentions', 'record_count']
    
    ax1.plot(daily_stats['date'], daily_stats['total_mentions'], marker='o', linewidth=2, 
             color='navy', markersize=3)
    ax1.fill_between(daily_stats['date'], daily_stats['total_mentions'], alpha=0.3, color='navy')
    ax1.set_ylabel('Total Mentions')
    ax1.set_title('Daily Mentions Over Time')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(daily_stats['date'], daily_stats['record_count'], marker='s', linewidth=2, 
             color='darkgreen', markersize=3)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Records per Day')
    ax2.set_title('Records per Day')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_daily_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved data_daily_trends.png")
    
    # Plot 5: Records per product
    print("5/5: Data sparsity analysis...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    records_per_product = df.groupby('product').size()
    ax1.hist(records_per_product.values, bins=40, color='purple', edgecolor='black', alpha=0.7)
    ax1.axvline(records_per_product.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {records_per_product.mean():.2f}')
    ax1.set_xlabel('Records per Product')
    ax1.set_ylabel('Number of Products')
    ax1.set_yscale('log')
    ax1.set_title('Data Distribution: Records per Product')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Data quality metrics
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    sparsity = (1 - len(df) / (df['product'].nunique() * (df['date'].max() - df['date'].min()).days)) * 100
    
    quality_text = f"""
    DATA QUALITY METRICS
    
    Records: {len(df):,}
    Products: {df['product'].nunique():,}
    Days: {(df['date'].max() - df['date'].min()).days}
    
    Completeness: {completeness:.2f}%
    Sparsity: {sparsity:.2f}%
    
    Avg Mentions: {df['mentions'].mean():.2f}
    Max Mentions: {df['mentions'].max():,}
    
    Avg Sentiment: {df['sentiment'].mean():.3f}
    Sentiment Std: {df['sentiment'].std():.3f}
    """
    
    ax2.text(0.1, 0.9, quality_text, fontsize=11, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             transform=ax2.transAxes)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_quality_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved data_quality_metrics.png")
    
    print("\n" + "="*70)
    print("[OK] DATA EXPLORATION COMPLETE!")
    print("="*70)
    print(f"\n[CHART] Visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for png_file in sorted(output_dir.glob('data_*.png')):
        print(f"   [OK] {png_file.name}")

if __name__ == '__main__':
    quick_explore()
