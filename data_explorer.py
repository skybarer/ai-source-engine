"""
Data Explorer: Comprehensive data analysis and visualizations
Provides insights into data distribution, quality, and characteristics
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

class DataExplorer:
    """Analyze and visualize e-commerce trend data"""

    def __init__(self):
        self.df = None
        self.output_dir = Path(PLOTS_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load processed trend data"""
        print("\n" + "="*70)
        print("DATA EXPLORATION & VISUALIZATION")
        print("="*70)
        
        try:
            self.df = pd.read_csv(PROCESSED_DATA_DIR / 'trend_data.csv')
            print(f"\n[OK] Loaded {len(self.df):,} records")
            print(f"[OK] Total unique products: {self.df['product'].nunique():,}")
            return True
        except FileNotFoundError:
            print(f"[FAIL] Data file not found: {PROCESSED_DATA_DIR / 'trend_data.csv'}")
            return False

    def print_data_summary(self):
        """Print comprehensive data summary"""
        print("\n" + "-"*70)
        print("DATA SUMMARY")
        print("-"*70)
        
        print(f"\n[CHART] Dataset Shape: {self.df.shape}")
        print(f"\n📅 Date Range:")
        print(f"   Start: {self.df['date'].min()}")
        print(f"   End:   {self.df['date'].max()}")
        print(f"   Duration: {(self.df['date'].max() - self.df['date'].min()).days} days")
        
        print(f"\n📦 Product Statistics:")
        print(f"   Total products: {self.df['product'].nunique():,}")
        print(f"   Records per product (avg): {len(self.df) / self.df['product'].nunique():.2f}")
        print(f"   Records per product (max): {self.df.groupby('product').size().max()}")
        print(f"   Records per product (min): {self.df.groupby('product').size().min()}")
        
        print(f"\n📈 Mentions Statistics:")
        print(f"   Total mentions: {self.df['mentions'].sum():,}")
        print(f"   Avg mentions/record: {self.df['mentions'].mean():.2f}")
        print(f"   Max mentions: {self.df['mentions'].max():,}")
        print(f"   Min mentions: {self.df['mentions'].min():.0f}")
        
        print(f"\n⭐ Sentiment Statistics:")
        print(f"   Mean sentiment: {self.df['sentiment'].mean():.3f}")
        print(f"   Std sentiment: {self.df['sentiment'].std():.3f}")
        print(f"   Min sentiment: {self.df['sentiment'].min():.3f}")
        print(f"   Max sentiment: {self.df['sentiment'].max():.3f}")

    def plot_mentions_distribution(self):
        """Plot distribution of mentions across products"""
        print("\n1️⃣  Creating mentions distribution plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Mentions Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(self.df['mentions'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Mentions per Record')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Mentions')
        axes[0, 0].set_yscale('log')
        
        # Box plot
        axes[0, 1].boxplot(self.df['mentions'])
        axes[0, 1].set_ylabel('Mentions')
        axes[0, 1].set_title('Box Plot of Mentions')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Product mentions
        product_mentions = self.df.groupby('product')['mentions'].sum().sort_values(ascending=False).head(15)
        axes[1, 0].barh(range(len(product_mentions)), product_mentions.values, color='coral')
        axes[1, 0].set_yticks(range(len(product_mentions)))
        axes[1, 0].set_yticklabels([name[:30] for name in product_mentions.index], fontsize=8)
        axes[1, 0].set_xlabel('Total Mentions')
        axes[1, 0].set_title('Top 15 Products by Mentions')
        axes[1, 0].invert_yaxis()
        
        # Cumulative
        sorted_mentions = self.df['mentions'].sort_values().values
        cumsum = np.cumsum(sorted_mentions) / np.sum(sorted_mentions) * 100
        axes[1, 1].plot(cumsum, linewidth=2, color='darkgreen')
        axes[1, 1].axhline(y=80, color='red', linestyle='--', label='80% threshold')
        axes[1, 1].axhline(y=20, color='orange', linestyle='--', label='20% threshold')
        axes[1, 1].set_xlabel('Records (sorted)')
        axes[1, 1].set_ylabel('Cumulative % of Mentions')
        axes[1, 1].set_title('Cumulative Mentions Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'data_mentions_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [OK] Saved to {output_path}")

    def plot_sentiment_analysis(self):
        """Plot sentiment statistics"""
        print("\n2️⃣  Creating sentiment analysis plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Sentiment Analysis', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(self.df['sentiment'], bins=40, color='mediumseagreen', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Sentiment Score (0-1)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Sentiment Scores')
        axes[0, 0].axvline(self.df['sentiment'].mean(), color='red', linestyle='--', 
                           label=f'Mean: {self.df["sentiment"].mean():.3f}')
        axes[0, 0].legend()
        
        # Box plot
        axes[0, 1].boxplot(self.df['sentiment'])
        axes[0, 1].set_ylabel('Sentiment Score')
        axes[0, 1].set_title('Sentiment Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sentiment bins
        sentiment_bins = pd.cut(self.df['sentiment'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
        sentiment_counts = sentiment_bins.value_counts().sort_index()
        axes[1, 0].bar(range(len(sentiment_counts)), sentiment_counts.values, color='skyblue', edgecolor='black')
        axes[1, 0].set_xticks(range(len(sentiment_counts)))
        axes[1, 0].set_xticklabels([f'{interval.left:.1f}-{interval.right:.1f}' for interval in sentiment_counts.index])
        axes[1, 0].set_xlabel('Sentiment Range')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Sentiment Distribution by Range')
        
        # Product average sentiment
        product_sentiment = self.df.groupby('product')['sentiment'].mean().sort_values(ascending=False).head(15)
        colors = plt.cm.RdYlGn(product_sentiment.values)
        axes[1, 1].barh(range(len(product_sentiment)), product_sentiment.values, color=colors)
        axes[1, 1].set_yticks(range(len(product_sentiment)))
        axes[1, 1].set_yticklabels([name[:30] for name in product_sentiment.index], fontsize=8)
        axes[1, 1].set_xlabel('Average Sentiment')
        axes[1, 1].set_title('Top 15 Products by Sentiment')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        output_path = self.output_dir / 'data_sentiment_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [OK] Saved to {output_path}")

    def plot_temporal_analysis(self):
        """Plot temporal patterns"""
        print("\n3️⃣  Creating temporal analysis plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Temporal Patterns Analysis', fontsize=16, fontweight='bold')
        
        # Daily mentions
        daily_mentions = self.df.groupby('date')['mentions'].agg(['sum', 'mean', 'count'])
        axes[0, 0].plot(daily_mentions.index, daily_mentions['sum'], marker='o', linewidth=2, 
                       color='navy', markersize=3, label='Total Mentions')
        axes[0, 0].fill_between(daily_mentions.index, daily_mentions['sum'], alpha=0.3, color='navy')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Total Mentions')
        axes[0, 0].set_title('Daily Mentions Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Records per day
        axes[0, 1].plot(daily_mentions.index, daily_mentions['count'], marker='s', linewidth=2, 
                       color='darkgreen', markersize=3, label='Record Count')
        axes[0, 1].fill_between(daily_mentions.index, daily_mentions['count'], alpha=0.3, color='darkgreen')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Records')
        axes[0, 1].set_title('Records per Day')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average mentions per day
        axes[1, 0].plot(daily_mentions.index, daily_mentions['mean'], marker='^', linewidth=2, 
                       color='purple', markersize=3, label='Avg Mentions')
        axes[1, 0].fill_between(daily_mentions.index, daily_mentions['mean'], alpha=0.3, color='purple')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Average Mentions')
        axes[1, 0].set_title('Average Mentions per Record per Day')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Records per product distribution
        records_per_product = self.df.groupby('product').size()
        axes[1, 1].hist(records_per_product.values, bins=50, color='darkorange', edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('Records per Product')
        axes[1, 1].set_ylabel('Number of Products')
        axes[1, 1].set_title('Distribution of Records per Product')
        axes[1, 1].set_yscale('log')
        axes[1, 1].axvline(records_per_product.mean(), color='red', linestyle='--', 
                          label=f'Mean: {records_per_product.mean():.2f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        output_path = self.output_dir / 'data_temporal_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [OK] Saved to {output_path}")

    def plot_data_quality(self):
        """Plot data quality metrics"""
        print("\n4️⃣  Creating data quality report...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Data Quality Assessment', fontsize=16, fontweight='bold')
        
        # Missing values
        missing = self.df.isnull().sum()
        axes[0, 0].bar(range(len(missing)), missing.values, color='crimson', alpha=0.7)
        axes[0, 0].set_xticks(range(len(missing)))
        axes[0, 0].set_xticklabels(missing.index)
        axes[0, 0].set_ylabel('Missing Count')
        axes[0, 0].set_title('Missing Values per Column')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Data type distribution
        dtype_counts = self.df.dtypes.value_counts()
        colors_pie = plt.cm.Set3(range(len(dtype_counts)))
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', colors=colors_pie)
        axes[0, 1].set_title('Data Types Distribution')
        
        # Duplicate check
        duplicate_products = self.df.groupby(['date', 'product']).size()
        duplicates_count = (duplicate_products > 1).sum()
        axes[1, 0].text(0.5, 0.7, f'Data Quality Metrics', ha='center', va='center', 
                       fontsize=14, fontweight='bold', transform=axes[1, 0].transAxes)
        
        quality_text = f"""
        [OK] Total Records: {len(self.df):,}
        [OK] Unique Products: {self.df['product'].nunique():,}
        [OK] Date Range: {(self.df['date'].max() - self.df['date'].min()).days} days
        [OK] Missing Values: {self.df.isnull().sum().sum()}
        [OK] Duplicate Date-Product: {duplicates_count}
        [OK] Completeness: {(1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100:.2f}%
        [OK] Data Sparsity: {(1 - len(self.df) / (self.df['product'].nunique() * (self.df['date'].max() - self.df['date'].min()).days)) * 100:.2f}%
        """
        
        axes[1, 0].text(0.1, 0.3, quality_text, ha='left', va='top', fontsize=10, 
                       fontfamily='monospace', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')
        
        # Data source breakdown (if available)
        if 'source' in self.df.columns:
            source_counts = self.df['source'].value_counts()
            colors_bar = plt.cm.Spectral(np.linspace(0, 1, len(source_counts)))
            axes[1, 1].bar(range(len(source_counts)), source_counts.values, color=colors_bar, edgecolor='black')
            axes[1, 1].set_xticks(range(len(source_counts)))
            axes[1, 1].set_xticklabels(source_counts.index)
            axes[1, 1].set_ylabel('Record Count')
            axes[1, 1].set_title('Records by Source')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'data_quality_report.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [OK] Saved to {output_path}")

    def plot_sample_product_trends(self):
        """Plot sample product trends"""
        print("\n5️⃣  Creating sample product trends...")
        
        # Get top 6 products by mentions
        top_products = self.df.groupby('product')['mentions'].sum().sort_values(ascending=False).head(6).index.tolist()
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Sample Product Trends (Top 6 by Total Mentions)', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        for idx, product in enumerate(top_products):
            product_data = self.df[self.df['product'] == product].sort_values('date')
            
            ax = axes[idx]
            ax.plot(product_data['date'], product_data['mentions'], marker='o', linewidth=2, 
                   color='navy', markersize=4, label='Mentions')
            ax2 = ax.twinx()
            ax2.plot(product_data['date'], product_data['sentiment'], marker='s', linewidth=2, 
                    color='coral', markersize=4, label='Sentiment')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Mentions', color='navy')
            ax2.set_ylabel('Sentiment', color='coral')
            ax.set_title(f'{product[:40]}...', fontweight='bold', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', labelcolor='navy')
            ax2.tick_params(axis='y', labelcolor='coral')
        
        plt.tight_layout()
        output_path = self.output_dir / 'sample_product_trends.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [OK] Saved to {output_path}")

    def generate_data_summary_table(self):
        """Generate summary statistics table"""
        print("\n6️⃣  Creating summary statistics table...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Summary statistics
        summary_data = {
            'Metric': [
                'Total Records',
                'Unique Products',
                'Date Range (days)',
                'Avg Records/Product',
                'Total Mentions',
                'Avg Mentions/Record',
                'Max Mentions/Record',
                'Min Mentions/Record',
                'Avg Sentiment',
                'Sentiment Std Dev',
                'Data Completeness',
                'Data Sparsity'
            ],
            'Value': [
                f"{len(self.df):,}",
                f"{self.df['product'].nunique():,}",
                f"{(self.df['date'].max() - self.df['date'].min()).days}",
                f"{len(self.df) / self.df['product'].nunique():.2f}",
                f"{self.df['mentions'].sum():,}",
                f"{self.df['mentions'].mean():.2f}",
                f"{self.df['mentions'].max():,}",
                f"{self.df['mentions'].min():.0f}",
                f"{self.df['sentiment'].mean():.4f}",
                f"{self.df['sentiment'].std():.4f}",
                f"{(1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100:.2f}%",
                f"{(1 - len(self.df) / (self.df['product'].nunique() * (self.df['date'].max() - self.df['date'].min()).days)) * 100:.2f}%"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                        cellLoc='center', loc='center', colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_df) + 1):
            for j in range(len(summary_df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        plt.title('Data Summary Statistics', fontsize=16, fontweight='bold', pad=20)
        
        output_path = self.output_dir / 'data_summary_table.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [OK] Saved to {output_path}")

    def run_all(self):
        """Run all analyses"""
        if not self.load_data():
            return
        
        self.print_data_summary()
        
        print("\n" + "-"*70)
        print("GENERATING VISUALIZATIONS")
        print("-"*70)
        
        self.plot_mentions_distribution()
        self.plot_sentiment_analysis()
        self.plot_temporal_analysis()
        self.plot_data_quality()
        self.plot_sample_product_trends()
        self.generate_data_summary_table()
        
        print("\n" + "="*70)
        print("[OK] DATA EXPLORATION COMPLETE!")
        print("="*70)
        print(f"\n[CHART] All visualizations saved to: {self.output_dir}")
        print("\nGenerated files:")
        for png_file in sorted(self.output_dir.glob('data_*.png')):
            print(f"   • {png_file.name}")
        for png_file in sorted(self.output_dir.glob('sample_*.png')):
            print(f"   • {png_file.name}")

if __name__ == '__main__':
    explorer = DataExplorer()
    explorer.run_all()
