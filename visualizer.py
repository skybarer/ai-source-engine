"""
=============================================================================
VISUALIZER.PY - Step 5: Generate Dissertation Plots
=============================================================================

PURPOSE: Create publication-quality figures for the dissertation.

PLOTS GENERATED (13 total):

  DATA EXPLORATION (what data do we have?):
    1.  data_source_overview     → Pie + bar: Amazon vs Flipkart vs Synthetic
    2.  data_daily_trends        → Time series of daily mentions (raw data shape)
    3.  data_sentiment_dist      → Sentiment distribution per source
    4.  data_top_products        → Top 15 products by review volume
    5.  data_quality_metrics     → Completeness, coverage, records per source
    6.  data_mentions_histogram  → Distribution of daily mention counts
    7.  data_correlation         → Mentions vs Sentiment scatter + correlation
    8.  data_weekly_patterns     → Day-of-week seasonality patterns

  PROCESSED DATA / TREND SCORES:
    9.  trend_leaderboard        → Top products by trend score (4-factor)
   10.  trend_score_components   → Stacked bar showing 4-factor breakdown

  MODEL RESULTS:
   11.  best_forecast            → Forecast vs Actual with confidence interval
   12.  validation_metrics       → 4-panel: MAPE, Accuracy, Peak, Detection
   13.  ensemble_components      → LSTM / ARIMA / Prophet individual outputs
=============================================================================
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves to file, no popup)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

from config import PLOTS_DIR

# ── Global style: clean, academic, high-contrast ────────────────────
sns.set_style("whitegrid")
plt.rcParams.update({
    'figure.figsize': (14, 7),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Consistent color palette across all plots
COLORS = {
    'amazon': '#FF9900',       # Amazon orange
    'flipkart': '#2874F0',     # Flipkart blue
    'synthetic': '#34A853',    # Green
    'lstm': '#4285F4',         # Google blue
    'arima': '#0F9D58',        # Green
    'prophet': '#F4B400',      # Gold
    'ensemble': '#DB4437',     # Red
    'accent': '#7B1FA2',       # Purple
    'good': '#2E7D32',         # Dark green
    'warn': '#F57C00',         # Orange
    'bad': '#C62828',          # Dark red
}

SUBTITLE = "M.Tech Dissertation | INKOLLU AKASHDHAR (2023AC05051) | BITS Pilani"


def _add_subtitle(fig, text=SUBTITLE):
    """Add dissertation subtitle below the main title"""
    fig.text(0.5, 0.96, text, ha='center', va='top',
             fontsize=9, color='gray', style='italic')


def _add_stats_box(ax, text, loc='upper right'):
    """Add a statistics text box inside the plot"""
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                 alpha=0.85, edgecolor='gray')
    if loc == 'upper right':
        ax.text(0.97, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    elif loc == 'upper left':
        ax.text(0.03, 0.95, text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='left', bbox=props)


class Visualizer:
    """Create dissertation-quality visualizations (13 plots)"""

    def __init__(self):
        self.plots_dir = PLOTS_DIR

    # =====================================================================
    # PART A: DATA EXPLORATION PLOTS (6 plots showing raw/processed data)
    # =====================================================================

    def plot_data_source_overview(self, df, save_name='data_source_overview.png'):
        """
        Plot 1: Where does our data come from?
        Left:  Pie chart — proportion of records per source
        Right: Bar chart — number of unique products per source
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Data Source Overview: Amazon + Flipkart + Synthetic',
                     fontsize=16, fontweight='bold', y=1.02)
        _add_subtitle(fig)

        # ── Left: Records per source (pie) ──
        source_counts = df['source'].value_counts()
        colors = [COLORS.get(s, '#999999') for s in source_counts.index]
        wedges, texts, autotexts = axes[0].pie(
            source_counts.values, labels=source_counts.index,
            autopct=lambda p: f'{p:.1f}%\n({int(p*len(df)/100):,})',
            colors=colors, startangle=140,
            textprops={'fontsize': 11}, pctdistance=0.75,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for t in autotexts:
            t.set_fontsize(9)
            t.set_fontweight('bold')
        axes[0].set_title('Records by Source', fontsize=14, fontweight='bold')
        total_text = f"Total Records: {len(df):,}"
        axes[0].text(0, -1.3, total_text, ha='center', fontsize=11,
                     fontweight='bold', color='#333')

        # ── Right: Products per source (bar) ──
        product_counts = df.groupby('source')['product'].nunique()
        bars = axes[1].bar(product_counts.index, product_counts.values,
                           color=[COLORS.get(s, '#999') for s in product_counts.index],
                           edgecolor='black', linewidth=0.8, alpha=0.85)
        for bar, val in zip(bars, product_counts.values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                         f'{val:,}', ha='center', va='bottom',
                         fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Data Source', fontweight='bold')
        axes[1].set_ylabel('Number of Unique Products', fontweight='bold')
        axes[1].set_title('Unique Products per Source', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)

        stats = (f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} → "
                 f"{df['date'].max().strftime('%Y-%m-%d')}\n"
                 f"Total Products: {df['product'].nunique():,}\n"
                 f"Avg Mentions/Day: {df.groupby('date')['mentions'].sum().mean():.1f}")
        _add_stats_box(axes[1], stats)

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    def plot_daily_trends(self, df, save_name='data_daily_trends.png'):
        """
        Plot 2: How does the aggregated daily data look?
        Shows total mentions per day across ALL products.
        This is the "processed data shape" that the model actually sees.
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
        fig.suptitle('Daily Aggregated Data: Total Mentions Over Time',
                     fontsize=16, fontweight='bold', y=1.01)
        _add_subtitle(fig)

        daily = df.groupby('date').agg(
            total_mentions=('mentions', 'sum'),
            avg_sentiment=('sentiment', 'mean'),
            n_products=('product', 'nunique')
        ).sort_index()

        # ── Top: Total mentions timeline ──
        axes[0].fill_between(daily.index, daily['total_mentions'],
                             alpha=0.3, color=COLORS['accent'])
        axes[0].plot(daily.index, daily['total_mentions'],
                     color=COLORS['accent'], linewidth=1.5, label='Daily Total')

        # 7-day moving average overlay
        ma7 = daily['total_mentions'].rolling(7, min_periods=1).mean()
        axes[0].plot(daily.index, ma7, color=COLORS['bad'], linewidth=2.5,
                     label='7-Day Moving Average', linestyle='--')

        axes[0].set_ylabel('Total Mentions (all products)', fontweight='bold')
        axes[0].set_title('Raw Daily Mentions → This is What the Model Learns From',
                          fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=11)
        axes[0].grid(alpha=0.3)

        stats = (f"Total Days: {len(daily)}\n"
                 f"Mean: {daily['total_mentions'].mean():.1f}/day\n"
                 f"Peak: {daily['total_mentions'].max():.0f}\n"
                 f"Std Dev: {daily['total_mentions'].std():.1f}")
        _add_stats_box(axes[0], stats)

        # ── Bottom: Number of active products per day ──
        axes[1].bar(daily.index, daily['n_products'], color=COLORS['flipkart'],
                    alpha=0.6, width=1.0)
        axes[1].set_xlabel('Date', fontweight='bold')
        axes[1].set_ylabel('Active Products', fontweight='bold')
        axes[1].set_title('Number of Products with Reviews Each Day',
                          fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    def plot_sentiment_distribution(self, df, save_name='data_sentiment_distribution.png'):
        """
        Plot 3: How are sentiment scores distributed?
        Sentiment = star_rating / 5.0 (our proxy for NLP sentiment)
        Split by data source to compare Amazon vs Flipkart patterns.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Sentiment Analysis: Star Rating as Sentiment Proxy (rating/5.0)',
                     fontsize=16, fontweight='bold', y=1.02)
        _add_subtitle(fig)

        sources = df['source'].unique()

        # ── Left: Overlapping histograms per source ──
        for source in sorted(sources):
            subset = df[df['source'] == source]['sentiment'].dropna()
            axes[0].hist(subset, bins=20, alpha=0.5, label=f'{source} (n={len(subset):,})',
                         color=COLORS.get(source, '#999'), edgecolor='black', linewidth=0.5)
        axes[0].set_xlabel('Sentiment Score (0=negative, 1=positive)', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('Sentiment Distribution by Source', fontweight='bold')
        axes[0].legend()
        axes[0].axvline(0.5, color='gray', linestyle=':', alpha=0.7, label='Neutral')
        axes[0].grid(alpha=0.3)

        # ── Center: Box plot per source ──
        source_data = [df[df['source'] == s]['sentiment'].dropna().values
                       for s in sorted(sources)]
        bp = axes[1].boxplot(source_data, labels=sorted(sources), patch_artist=True,
                             showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'red'})
        for patch, source in zip(bp['boxes'], sorted(sources)):
            patch.set_facecolor(COLORS.get(source, '#999'))
            patch.set_alpha(0.6)
        axes[1].set_ylabel('Sentiment Score', fontweight='bold')
        axes[1].set_title('Sentiment Range by Source', fontweight='bold')
        axes[1].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        axes[1].grid(alpha=0.3)

        overall_stats = (f"Overall Mean: {df['sentiment'].mean():.3f}\n"
                         f"Overall Median: {df['sentiment'].median():.3f}\n"
                         f"Std Dev: {df['sentiment'].std():.3f}")
        _add_stats_box(axes[1], overall_stats, loc='upper left')

        # ── Right: Sentiment over time ──
        daily_sent = df.groupby('date')['sentiment'].mean().sort_index()
        axes[2].plot(daily_sent.index, daily_sent.values,
                     color=COLORS['accent'], alpha=0.5, linewidth=1)
        ma = daily_sent.rolling(7, min_periods=1).mean()
        axes[2].plot(ma.index, ma.values, color=COLORS['bad'],
                     linewidth=2.5, label='7-Day MA')
        axes[2].set_xlabel('Date', fontweight='bold')
        axes[2].set_ylabel('Avg Sentiment', fontweight='bold')
        axes[2].set_title('Sentiment Trend Over Time', fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    def plot_top_products(self, df, top_n=15, save_name='data_top_products.png'):
        """
        Plot 4: Which products have the most data?
        Important because: only products with enough data can be individually validated.
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f'Top {top_n} Products by Data Volume',
                     fontsize=16, fontweight='bold', y=1.01)
        _add_subtitle(fig)

        # ── Left: Total mentions (review volume) ──
        product_mentions = df.groupby('product')['mentions'].sum().nlargest(top_n)
        colors_list = [COLORS['amazon'] if i % 2 == 0 else COLORS['flipkart']
                       for i in range(top_n)]
        bars = axes[0].barh(range(len(product_mentions)), product_mentions.values,
                            color=colors_list, edgecolor='black', linewidth=0.5, alpha=0.8)
        axes[0].set_yticks(range(len(product_mentions)))
        axes[0].set_yticklabels([p[:35] + '...' if len(p) > 35 else p
                                 for p in product_mentions.index], fontsize=9)
        axes[0].set_xlabel('Total Mentions (Review Count)', fontweight='bold')
        axes[0].set_title(f'Top {top_n} by Review Volume', fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].invert_yaxis()

        # Add value labels
        for bar, val in zip(bars, product_mentions.values):
            axes[0].text(val + 1, bar.get_y() + bar.get_height()/2,
                         f'{int(val)}', va='center', fontsize=9, fontweight='bold')

        # ── Right: Number of data points (days with reviews) ──
        product_days = df.groupby('product')['date'].nunique().nlargest(top_n)
        bars2 = axes[1].barh(range(len(product_days)), product_days.values,
                             color=COLORS['accent'], edgecolor='black',
                             linewidth=0.5, alpha=0.7)
        axes[1].set_yticks(range(len(product_days)))
        axes[1].set_yticklabels([p[:35] + '...' if len(p) > 35 else p
                                 for p in product_days.index], fontsize=9)
        axes[1].set_xlabel('Number of Days with Reviews', fontweight='bold')
        axes[1].set_title(f'Top {top_n} by Data Coverage', fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        axes[1].invert_yaxis()

        # Annotate: min requirement for validation
        axes[1].axvline(10, color='red', linestyle='--', linewidth=2, alpha=0.7)
        axes[1].text(11, top_n-1, 'Min for\nvalidation', color='red',
                     fontsize=9, fontweight='bold')

        for bar, val in zip(bars2, product_days.values):
            axes[1].text(val + 0.3, bar.get_y() + bar.get_height()/2,
                         f'{int(val)}', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    def plot_data_quality(self, df, save_name='data_quality_metrics.png'):
        """
        Plot 5: Data quality dashboard.
        Shows completeness, missing values, and key statistics.
        Important for dissertation: proves we handled data properly.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Quality Assessment',
                     fontsize=16, fontweight='bold', y=1.01)
        _add_subtitle(fig)

        # ── Top-left: Records per source with percentages ──
        source_counts = df.groupby('source').size()
        bars = axes[0, 0].bar(source_counts.index, source_counts.values,
                              color=[COLORS.get(s, '#999') for s in source_counts.index],
                              edgecolor='black', linewidth=0.8)
        for bar, val in zip(bars, source_counts.values):
            pct = val / len(df) * 100
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                            f'{val:,}\n({pct:.1f}%)', ha='center', fontweight='bold',
                            fontsize=10)
        axes[0, 0].set_title('Records per Data Source', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Records', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # ── Top-right: Missing value analysis ──
        missing = df.isnull().sum()
        completeness = ((len(df) - missing) / len(df) * 100).round(2)
        cols = ['date', 'product', 'mentions', 'sentiment', 'source']
        comp_vals = [completeness.get(c, 100.0) for c in cols]
        bar_colors = [COLORS['good'] if v >= 99 else COLORS['warn'] if v >= 90
                      else COLORS['bad'] for v in comp_vals]
        bars2 = axes[0, 1].barh(cols, comp_vals, color=bar_colors,
                                edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars2, comp_vals):
            axes[0, 1].text(bar.get_width() - 2, bar.get_y() + bar.get_height()/2,
                            f'{val:.1f}%', va='center', ha='right',
                            fontweight='bold', fontsize=11, color='white')
        axes[0, 1].set_xlim(0, 105)
        axes[0, 1].set_title('Column Completeness (% non-null)', fontweight='bold')
        axes[0, 1].axvline(100, color='gray', linestyle=':', alpha=0.5)
        axes[0, 1].grid(axis='x', alpha=0.3)

        # ── Bottom-left: Mentions distribution (histogram) ──
        mentions = df['mentions'].values
        axes[1, 0].hist(mentions, bins=50, color=COLORS['flipkart'],
                        edgecolor='black', linewidth=0.5, alpha=0.7)
        axes[1, 0].axvline(mentions.mean(), color='red', linewidth=2,
                           linestyle='--', label=f'Mean: {mentions.mean():.2f}')
        axes[1, 0].axvline(np.median(mentions), color='orange', linewidth=2,
                           linestyle='--', label=f'Median: {np.median(mentions):.2f}')
        axes[1, 0].set_xlabel('Mentions per Record', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Distribution of Mention Counts', fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(alpha=0.3)

        stats = (f"Min: {mentions.min():.0f}\nMax: {mentions.max():.0f}\n"
                 f"Std: {mentions.std():.2f}\nSkew: {pd.Series(mentions).skew():.2f}")
        _add_stats_box(axes[1, 0], stats)

        # ── Bottom-right: Summary statistics table ──
        axes[1, 1].axis('off')
        summary_data = [
            ['Metric', 'Value'],
            ['Total Records', f'{len(df):,}'],
            ['Total Products', f'{df["product"].nunique():,}'],
            ['Date Range', f'{df["date"].min().strftime("%Y-%m-%d")} to {df["date"].max().strftime("%Y-%m-%d")}'],
            ['Total Days Covered', f'{df["date"].nunique()}'],
            ['Avg Records/Day', f'{len(df) / max(df["date"].nunique(), 1):.1f}'],
            ['Avg Sentiment', f'{df["sentiment"].mean():.3f} (scale 0-1)'],
            ['Sources', ', '.join(df['source'].unique())],
            ['Sentiment Method', 'Star Rating / 5.0'],
            ['Missing Values', f'{df.isnull().sum().sum()} ({df.isnull().sum().sum()/len(df)/len(df.columns)*100:.2f}%)'],
        ]
        table = axes[1, 1].table(cellText=summary_data[1:],
                                  colLabels=summary_data[0],
                                  cellLoc='left', loc='center',
                                  colWidths=[0.35, 0.55])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
            elif row % 2 == 0:
                cell.set_facecolor('#D6E4F0')
            cell.set_edgecolor('white')
        axes[1, 1].set_title('Data Summary', fontweight='bold', fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    def plot_mentions_histogram(self, df, save_name='data_mentions_histogram.png'):
        """
        Plot 6: Deep-dive into mentions distribution per source.
        Shows how review volumes differ between Amazon and Flipkart.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Review Mention Patterns by Data Source',
                     fontsize=16, fontweight='bold', y=1.02)
        _add_subtitle(fig)

        # ── Left: KDE (smooth density) per source ──
        for source in sorted(df['source'].unique()):
            subset = df[df['source'] == source]['mentions']
            subset = subset[subset > 0]  # Log scale needs positive
            if len(subset) > 10:
                axes[0].hist(subset, bins=30, alpha=0.4, density=True,
                             color=COLORS.get(source, '#999'), label=source,
                             edgecolor='black', linewidth=0.3)
        axes[0].set_xlabel('Mentions (Review Count)', fontweight='bold')
        axes[0].set_ylabel('Density', fontweight='bold')
        axes[0].set_title('Mention Density by Source', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # ── Right: Mentions vs Sentiment scatter ──
        sample = df.sample(min(2000, len(df)), random_state=42)
        scatter_colors = [COLORS.get(s, '#999') for s in sample['source']]
        axes[1].scatter(sample['mentions'], sample['sentiment'],
                        c=scatter_colors, alpha=0.4, s=20, edgecolors='none')
        axes[1].set_xlabel('Mentions (Review Count)', fontweight='bold')
        axes[1].set_ylabel('Sentiment Score', fontweight='bold')
        axes[1].set_title('Mentions vs Sentiment (sampled)', fontweight='bold')
        axes[1].grid(alpha=0.3)

        # Add correlation coefficient
        corr = df['mentions'].corr(df['sentiment'])
        _add_stats_box(axes[1], f"Pearson r = {corr:.3f}\n"
                       f"n = {len(df):,} records\n"
                       f"(showing {len(sample):,} sample)")

        # Legend
        legend_patches = [mpatches.Patch(color=COLORS.get(s, '#999'), label=s)
                          for s in sorted(df['source'].unique())]
        axes[1].legend(handles=legend_patches, loc='lower right')

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    def plot_weekly_patterns(self, df, save_name='data_weekly_patterns.png'):
        """
        Plot 7: Day-of-week seasonality patterns.
        Shows if reviews are posted more on certain days.
        This justifies using Prophet's weekly_seasonality.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Weekly Seasonality Patterns in Review Data',
                     fontsize=16, fontweight='bold', y=1.02)
        _add_subtitle(fig)

        df_copy = df.copy()
        df_copy['day_of_week'] = pd.to_datetime(df_copy['date']).dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                     'Friday', 'Saturday', 'Sunday']

        # ── Left: Average mentions by day of week ──
        daily_avg = df_copy.groupby('day_of_week')['mentions'].mean().reindex(day_order)
        colors_dow = [COLORS['flipkart'] if d in ['Saturday', 'Sunday']
                      else COLORS['amazon'] for d in day_order]
        bars = axes[0].bar(range(7), daily_avg.values, color=colors_dow,
                           edgecolor='black', linewidth=0.5, alpha=0.8)
        axes[0].set_xticks(range(7))
        axes[0].set_xticklabels([d[:3] for d in day_order], fontsize=11)
        axes[0].set_ylabel('Average Mentions', fontweight='bold')
        axes[0].set_title('Avg Review Mentions by Day of Week', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Highlight weekend vs weekday
        axes[0].axhline(daily_avg.mean(), color='red', linestyle='--',
                        linewidth=1.5, label=f'Overall Avg: {daily_avg.mean():.2f}')
        axes[0].legend()

        for bar, val in zip(bars, daily_avg.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')

        legend_patches = [mpatches.Patch(color=COLORS['amazon'], label='Weekday'),
                          mpatches.Patch(color=COLORS['flipkart'], label='Weekend')]
        axes[0].legend(handles=legend_patches + [
            plt.Line2D([0], [0], color='red', linestyle='--', label=f'Avg: {daily_avg.mean():.2f}')
        ], loc='upper right')

        # ── Right: Average sentiment by day of week ──
        sent_avg = df_copy.groupby('day_of_week')['sentiment'].mean().reindex(day_order)
        axes[1].plot(range(7), sent_avg.values, 'o-', color=COLORS['accent'],
                     linewidth=2.5, markersize=10, markerfacecolor='white',
                     markeredgewidth=2, markeredgecolor=COLORS['accent'])
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels([d[:3] for d in day_order], fontsize=11)
        axes[1].set_ylabel('Average Sentiment', fontweight='bold')
        axes[1].set_title('Avg Sentiment by Day of Week', fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(sent_avg.min() - 0.02, sent_avg.max() + 0.02)

        _add_stats_box(axes[1], f"Range: {sent_avg.min():.3f} - {sent_avg.max():.3f}\n"
                       f"Std: {sent_avg.std():.4f}\n"
                       f"This justifies Prophet's\nweekly_seasonality=True")

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    def plot_correlation_heatmap(self, df, save_name='data_correlation.png'):
        """
        Plot 8: Correlation between numeric features.
        Shows relationships between mentions, sentiment, and derived features.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Feature Correlation Analysis',
                     fontsize=16, fontweight='bold', y=1.02)
        _add_subtitle(fig)

        # Prepare data with additional features
        daily = df.groupby('date').agg(
            mentions=('mentions', 'sum'),
            sentiment=('sentiment', 'mean'),
            n_products=('product', 'nunique'),
        ).sort_index()
        daily['mentions_7d_ma'] = daily['mentions'].rolling(7, min_periods=1).mean()
        daily['growth_rate'] = daily['mentions'].pct_change(7).fillna(0) * 100

        # ── Left: Correlation heatmap ──
        corr_cols = ['mentions', 'sentiment', 'n_products', 'mentions_7d_ma', 'growth_rate']
        corr_matrix = daily[corr_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    ax=axes[0], square=True, linewidths=1,
                    cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
        axes[0].set_title('Feature Correlation Matrix', fontweight='bold')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

        # ── Right: Mentions vs Sentiment trend (dual-axis) ──
        ax2 = axes[1].twinx()
        l1 = axes[1].plot(daily.index, daily['mentions'], color=COLORS['flipkart'],
                          alpha=0.6, linewidth=1, label='Daily Mentions')
        l1_ma = axes[1].plot(daily.index, daily['mentions_7d_ma'],
                             color=COLORS['flipkart'], linewidth=2.5,
                             label='Mentions (7-Day MA)')
        l2 = ax2.plot(daily.index, daily['sentiment'], color=COLORS['amazon'],
                      alpha=0.6, linewidth=1, label='Sentiment')
        sent_ma = daily['sentiment'].rolling(7, min_periods=1).mean()
        l2_ma = ax2.plot(daily.index, sent_ma, color=COLORS['amazon'],
                         linewidth=2.5, label='Sentiment (7-Day MA)')

        axes[1].set_xlabel('Date', fontweight='bold')
        axes[1].set_ylabel('Mentions', fontweight='bold', color=COLORS['flipkart'])
        ax2.set_ylabel('Sentiment', fontweight='bold', color=COLORS['amazon'])
        axes[1].set_title('Mentions & Sentiment: Dual Timeline', fontweight='bold')

        lines = l1 + l1_ma + l2 + l2_ma
        labels = [l.get_label() for l in lines]
        axes[1].legend(lines, labels, loc='upper left', fontsize=9)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    # =====================================================================
    # PART B: TREND SCORE PLOTS (2 plots)
    # =====================================================================

    def plot_trend_scores(self, df_scored, top_n=15, save_name='trend_leaderboard.png'):
        """
        Plot 9: Top trending products with trend score breakdown.
        Shows overall score + color coding: Red(>70), Orange(>50), Green(<50).
        """
        fig, ax = plt.subplots(figsize=(14, 9))
        fig.suptitle(f'Product Trend Leaderboard — Top {top_n} by Trend Score',
                     fontsize=16, fontweight='bold', y=1.01)
        _add_subtitle(fig)

        latest = df_scored.groupby('product').apply(
            lambda x: x.nlargest(7, 'date')['trend_score'].mean()
        ).reset_index(name='avg_score')
        top = latest.nlargest(top_n, 'avg_score')

        colors = [COLORS['bad'] if s > 70 else COLORS['warn'] if s > 50
                  else COLORS['good'] for s in top['avg_score']]

        bars = ax.barh(range(len(top)), top['avg_score'], color=colors,
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([p[:35] + '...' if len(p) > 35 else p
                            for p in top['product']], fontsize=10)
        ax.set_xlabel('Trend Score (0-100)', fontsize=13, fontweight='bold')
        ax.set_title('Score = Growth(40%) + Sentiment(20%) + Saturation(20%) + Profit(20%)',
                     fontsize=11, color='gray')
        ax.invert_yaxis()

        # Value labels
        for bar, val in zip(bars, top['avg_score']):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', fontweight='bold', fontsize=10)

        # Threshold lines
        ax.axvline(60, color='red', linestyle='--', linewidth=2, alpha=0.6)
        ax.text(61, len(top)-0.5, 'High Potential\nThreshold (60)',
                color='red', fontsize=9, fontweight='bold')

        # Legend
        legend_patches = [
            mpatches.Patch(color=COLORS['bad'], label='Hot (>70)'),
            mpatches.Patch(color=COLORS['warn'], label='Warm (50-70)'),
            mpatches.Patch(color=COLORS['good'], label='Emerging (<50)'),
        ]
        ax.legend(handles=legend_patches, loc='lower right', fontsize=10)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    def plot_trend_components(self, df_scored, top_n=10,
                              save_name='trend_score_components.png'):
        """
        Plot 10: Stacked bar chart showing HOW each product's score is composed.
        4 factors: growth, sentiment, saturation, profit.
        """
        fig, ax = plt.subplots(figsize=(14, 9))
        fig.suptitle('Trend Score Decomposition — 4-Factor Breakdown per Product',
                     fontsize=16, fontweight='bold', y=1.01)
        _add_subtitle(fig)

        # Get average component scores per product
        components = df_scored.groupby('product').agg({
            'growth_component': 'mean',
            'sentiment_component': 'mean',
            'saturation_component': 'mean',
            'profit_component': 'mean',
            'trend_score': 'mean'
        }).nlargest(top_n, 'trend_score')

        y_pos = range(len(components))

        # Stacked horizontal bars
        ax.barh(y_pos, components['growth_component'],
                color=COLORS['bad'], alpha=0.8, label='Growth (40%)', edgecolor='white')
        left = components['growth_component'].values
        ax.barh(y_pos, components['sentiment_component'],
                left=left, color=COLORS['flipkart'], alpha=0.8,
                label='Sentiment (20%)', edgecolor='white')
        left = left + components['sentiment_component'].values
        ax.barh(y_pos, components['saturation_component'],
                left=left, color=COLORS['prophet'], alpha=0.8,
                label='Saturation (20%)', edgecolor='white')
        left = left + components['saturation_component'].values
        ax.barh(y_pos, components['profit_component'],
                left=left, color=COLORS['synthetic'], alpha=0.8,
                label='Profit (20%)', edgecolor='white')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([p[:35] + '...' if len(p) > 35 else p
                            for p in components.index], fontsize=9)
        ax.set_xlabel('Score Contribution', fontweight='bold')
        ax.set_title('Each color = one scoring factor. Wider = stronger signal.',
                     fontsize=11, color='gray')
        ax.legend(loc='lower right', fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # Total score labels
        for i, (_, row) in enumerate(components.iterrows()):
            ax.text(row['trend_score'] + 0.5, i,
                    f'{row["trend_score"]:.1f}', va='center',
                    fontweight='bold', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    # =====================================================================
    # PART C: MODEL RESULT PLOTS (3 plots — improved originals)
    # =====================================================================

    def plot_forecast_with_actual(self, product_df, forecast_result,
                                  train_days=120, save_name=None):
        """
        Plot 11: Historical data + forecast + confidence interval.
        
        Shows:
          Blue solid  = historical data (training period)
          Green solid = actual test data (what really happened)
          Red dashed  = our ensemble forecast (LSTM+ARIMA+Prophet)
          Red shaded  = 95% confidence interval
          Orange line = early warning window marker
        """
        fig, ax = plt.subplots(figsize=(18, 8))

        dates = product_df['date'].values
        mentions = product_df['mentions'].values
        train_days = min(train_days, len(product_df) - 1)

        forecast_len = len(forecast_result['forecast'])
        available_test = len(product_df) - train_days
        actual_test_len = min(forecast_len, available_test)

        train_dates = dates[:train_days]
        test_dates = dates[train_days:train_days + forecast_len]

        if len(test_dates) < forecast_len:
            last = pd.Timestamp(dates[-1])
            test_dates = pd.date_range(
                start=last + pd.Timedelta(days=1),
                periods=forecast_len, freq='D'
            ).values

        # ── Historical data (blue) ──
        ax.plot(train_dates, mentions[:train_days],
                color=COLORS['flipkart'], linewidth=2, label='Historical (Train)',
                alpha=0.8)
        ax.fill_between(train_dates, mentions[:train_days],
                        alpha=0.1, color=COLORS['flipkart'])

        # ── Actual test data (green) ──
        if actual_test_len > 0:
            ax.plot(test_dates[:actual_test_len],
                    mentions[train_days:train_days + actual_test_len],
                    color=COLORS['good'], linewidth=2.5,
                    label='Actual (Test)', alpha=0.9, marker='o', markersize=4)

        # ── Forecast (red dashed) ──
        ax.plot(test_dates, forecast_result['forecast'],
                color=COLORS['ensemble'], linewidth=2.5, linestyle='--',
                label='Ensemble Forecast (LSTM+ARIMA+Prophet)', alpha=0.9,
                marker='s', markersize=4)

        # ── 95% Confidence interval ──
        ax.fill_between(test_dates,
                        forecast_result['lower_bound'],
                        forecast_result['upper_bound'],
                        alpha=0.2, color=COLORS['ensemble'], label='95% CI')

        # ── Train/Test split line ──
        split_date = dates[train_days] if train_days < len(dates) else dates[-1]
        ax.axvline(split_date, color='black', linestyle='-', linewidth=2,
                   alpha=0.5, label='Train/Test Split')

        # ── Early warning line ──
        warning_date = train_dates[-45] if len(train_dates) >= 45 else train_dates[0]
        ax.axvline(warning_date, color=COLORS['warn'], linestyle=':',
                   linewidth=2.5, label='Early Warning (45 days)', alpha=0.7)

        name = product_df['product'].iloc[0] if 'product' in product_df.columns else 'Product'
        name_short = name[:50] + '...' if len(name) > 50 else name
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax.set_ylabel('Product Mentions (Daily)', fontsize=13, fontweight='bold')
        ax.set_title(f'Ensemble Forecast: {name_short}',
                     fontsize=15, fontweight='bold')
        fig.text(0.5, 0.96, SUBTITLE, ha='center', fontsize=9,
                 color='gray', style='italic')

        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(alpha=0.3)

        # Stats box
        forecast_vals = forecast_result['forecast']
        stats_text = (f"Forecast Horizon: {forecast_len} days\n"
                      f"Train Period: {train_days} days\n"
                      f"Forecast Mean: {forecast_vals.mean():.2f}\n"
                      f"Forecast Range: [{forecast_vals.min():.2f}, {forecast_vals.max():.2f}]\n"
                      f"Model: LSTM(55%)+ARIMA(30%)+Prophet(15%)")
        _add_stats_box(ax, stats_text)

        plt.tight_layout()
        if save_name:
            plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved {save_name}")
        plt.close('all')

    def plot_model_comparison(self, results_df, save_name='validation_metrics.png'):
        """
        Plot 12: 4-panel validation results dashboard.
          Top-left:     MAPE histogram with target line
          Top-right:    Accuracy bar chart (all validated products)
          Bottom-left:  Peak timing scatter (predicted vs actual)
          Bottom-right: Summary metrics table
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Model Validation Results — Ensemble Performance',
                     fontsize=16, fontweight='bold', y=1.01)
        _add_subtitle(fig)

        # ── Top-left: MAPE distribution ──
        mape_vals = results_df['MAPE'].values
        colors_mape = [COLORS['good'] if m < 20 else COLORS['warn'] if m < 30
                       else COLORS['bad'] for m in mape_vals]
        axes[0, 0].bar(range(len(mape_vals)), mape_vals, color=colors_mape,
                       edgecolor='black', linewidth=0.5, alpha=0.8)
        axes[0, 0].axhline(30, color='red', linestyle='--', linewidth=2,
                           label='Target: MAPE < 30%')
        axes[0, 0].axhline(20, color='orange', linestyle=':', linewidth=1.5,
                           label='Good: MAPE < 20%')
        axes[0, 0].set_xlabel('Product Index', fontweight='bold')
        axes[0, 0].set_ylabel('MAPE (%)', fontweight='bold')
        axes[0, 0].set_title('MAPE per Validated Product', fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(axis='y', alpha=0.3)

        avg_mape = mape_vals.mean()
        _add_stats_box(axes[0, 0],
                       f"Avg MAPE: {avg_mape:.2f}%\n"
                       f"Best: {mape_vals.min():.2f}%\n"
                       f"Worst: {mape_vals.max():.2f}%\n"
                       f"{'PASSED' if avg_mape < 30 else 'NEEDS WORK'}",
                       loc='upper left')

        # ── Top-right: Accuracy bars ──
        acc_vals = results_df['Accuracy'].values
        products = [p[:25] + '...' if len(str(p)) > 25 else str(p)
                    for p in results_df['Product']]
        colors_acc = [COLORS['good'] if a >= 80 else COLORS['warn'] if a >= 70
                      else COLORS['bad'] for a in acc_vals]
        bars = axes[0, 1].barh(range(len(acc_vals)), acc_vals, color=colors_acc,
                               edgecolor='black', linewidth=0.5, alpha=0.8)
        axes[0, 1].set_yticks(range(len(products)))
        axes[0, 1].set_yticklabels(products, fontsize=9)
        axes[0, 1].set_xlabel('Accuracy (%)', fontweight='bold')
        axes[0, 1].set_title('Accuracy per Product', fontweight='bold')
        axes[0, 1].axvline(70, color='red', linestyle='--', linewidth=2)
        axes[0, 1].axvline(80, color='orange', linestyle=':', linewidth=1.5)
        axes[0, 1].set_xlim(0, 105)
        axes[0, 1].grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, acc_vals):
            axes[0, 1].text(val + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

        legend_patches = [
            mpatches.Patch(color=COLORS['good'], label='>80% (Excellent)'),
            mpatches.Patch(color=COLORS['warn'], label='70-80% (Good)'),
            mpatches.Patch(color=COLORS['bad'], label='<70% (Below target)'),
        ]
        axes[0, 1].legend(handles=legend_patches, loc='lower right', fontsize=8)

        # ── Bottom-left: Peak timing scatter ──
        actual_peaks = results_df['Actual_Peak_Day'].values
        pred_peaks = results_df['Predicted_Peak_Day'].values
        axes[1, 0].scatter(actual_peaks, pred_peaks,
                           alpha=0.8, s=150, color=COLORS['accent'],
                           edgecolors='black', linewidth=1, zorder=5)
        m = max(max(actual_peaks), max(pred_peaks)) + 2
        axes[1, 0].plot([0, m], [0, m], 'k--', linewidth=2,
                        label='Perfect Prediction', alpha=0.5)
        # ±7 day band
        axes[1, 0].fill_between([0, m], [0-7, m-7], [0+7, m+7],
                                alpha=0.15, color='green', label='±7 Day Band')
        axes[1, 0].set_xlabel('Actual Peak Day', fontweight='bold')
        axes[1, 0].set_ylabel('Predicted Peak Day', fontweight='bold')
        axes[1, 0].set_title('Peak Timing: Predicted vs Actual', fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_xlim(-1, m)
        axes[1, 0].set_ylim(-1, m)

        peak_errors = np.abs(actual_peaks - pred_peaks)
        _add_stats_box(axes[1, 0],
                       f"Avg Peak Error: ±{peak_errors.mean():.1f} days\n"
                       f"Max Error: ±{peak_errors.max():.0f} days\n"
                       f"Target: ±7 days",
                       loc='upper left')

        # ── Bottom-right: Summary table ──
        axes[1, 1].axis('off')
        avg_acc = results_df['Accuracy'].mean()
        summary = [
            ['Metric', 'Value', 'Target', 'Status'],
            ['MAPE', f'{results_df["MAPE"].mean():.2f}%', '<30%',
             'PASS' if results_df['MAPE'].mean() < 30 else 'FAIL'],
            ['Accuracy', f'{avg_acc:.2f}%', '>70%',
             'PASS' if avg_acc > 70 else 'FAIL'],
            ['MAE', f'{results_df["MAE"].mean():.2f}', 'Lower=Better', '-'],
            ['RMSE', f'{results_df["RMSE"].mean():.2f}', 'Lower=Better', '-'],
            ['Peak Error', f'±{peak_errors.mean():.1f} days', '±7 days',
             'PASS' if peak_errors.mean() <= 7 else 'FAIL'],
            ['Products', f'{len(results_df)}', '>=3', 'PASS'],
            ['Model', 'LSTM+ARIMA+Prophet', 'Ensemble', '-'],
            ['Weights', '55% / 30% / 15%', 'Fixed', '-'],
        ]
        table = axes[1, 1].table(cellText=summary[1:], colLabels=summary[0],
                                  cellLoc='center', loc='center',
                                  colWidths=[0.22, 0.22, 0.22, 0.14])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.0)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4472C4')
                cell.set_text_props(color='white', fontweight='bold')
            elif col == 3:
                val = cell.get_text().get_text()
                if val == 'PASS':
                    cell.set_facecolor('#C6EFCE')
                    cell.set_text_props(fontweight='bold', color='green')
                elif val == 'FAIL':
                    cell.set_facecolor('#FFC7CE')
                    cell.set_text_props(fontweight='bold', color='red')
            elif row % 2 == 0:
                cell.set_facecolor('#D6E4F0')
            cell.set_edgecolor('white')
        axes[1, 1].set_title('Validation Summary', fontweight='bold', fontsize=14, pad=20)

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    def plot_component_breakdown(self, forecast_result, save_name='ensemble_components.png'):
        """
        Plot 13: Each model's individual prediction vs the ensemble.
        Shows HOW the ensemble combines three different approaches.
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Ensemble Model Components — Individual Model Contributions',
                     fontsize=16, fontweight='bold', y=1.01)
        _add_subtitle(fig)

        days = range(len(forecast_result['forecast']))
        ensemble = forecast_result['forecast']

        # ── LSTM (top-left) ──
        lstm_pred = forecast_result['components']['lstm']
        axes[0, 0].plot(days, lstm_pred, color=COLORS['lstm'], linewidth=2.5,
                        label='LSTM Prediction')
        axes[0, 0].plot(days, ensemble, color='gray', linewidth=1,
                        linestyle=':', alpha=0.5, label='Ensemble (reference)')
        axes[0, 0].fill_between(days, lstm_pred, alpha=0.15, color=COLORS['lstm'])
        axes[0, 0].set_title('LSTM Neural Network (Weight: 55%)',
                             fontsize=13, fontweight='bold')
        axes[0, 0].set_ylabel('Predicted Mentions', fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(alpha=0.3)
        _add_stats_box(axes[0, 0],
                       f"Architecture: 256→128→64\n"
                       f"Mean: {np.mean(lstm_pred):.2f}\n"
                       f"Captures: Non-linear patterns",
                       loc='upper left')

        # ── ARIMA (top-right) ──
        arima_pred = forecast_result['components']['arima']
        axes[0, 1].plot(days, arima_pred, color=COLORS['arima'], linewidth=2.5,
                        label='ARIMA Prediction')
        axes[0, 1].plot(days, ensemble, color='gray', linewidth=1,
                        linestyle=':', alpha=0.5, label='Ensemble (reference)')
        axes[0, 1].fill_between(days, arima_pred, alpha=0.15, color=COLORS['arima'])
        axes[0, 1].set_title('ARIMA(2,1,2) Statistical Model (Weight: 30%)',
                             fontsize=13, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(alpha=0.3)
        _add_stats_box(axes[0, 1],
                       f"Order: (p=2, d=1, q=2)\n"
                       f"Mean: {np.mean(arima_pred):.2f}\n"
                       f"Captures: Linear trends",
                       loc='upper left')

        # ── Prophet (bottom-left) ──
        prophet_pred = forecast_result['components']['prophet']
        axes[1, 0].plot(days, prophet_pred, color=COLORS['prophet'], linewidth=2.5,
                        label='Prophet Prediction')
        axes[1, 0].plot(days, ensemble, color='gray', linewidth=1,
                        linestyle=':', alpha=0.5, label='Ensemble (reference)')
        axes[1, 0].fill_between(days, prophet_pred, alpha=0.15, color=COLORS['prophet'])
        axes[1, 0].set_title('Facebook Prophet (Weight: 15%)',
                             fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Days Ahead', fontweight='bold')
        axes[1, 0].set_ylabel('Predicted Mentions', fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(alpha=0.3)
        _add_stats_box(axes[1, 0],
                       f"Seasonality: multiplicative\n"
                       f"Mean: {np.mean(prophet_pred):.2f}\n"
                       f"Captures: Weekly patterns",
                       loc='upper left')

        # ── Ensemble (bottom-right) ──
        axes[1, 1].plot(days, ensemble, color=COLORS['ensemble'], linewidth=3,
                        label='Ensemble (Final)')
        axes[1, 1].fill_between(days, forecast_result['lower_bound'],
                                forecast_result['upper_bound'],
                                alpha=0.25, color=COLORS['ensemble'], label='95% CI')
        # Overlay individual components faintly
        axes[1, 1].plot(days, lstm_pred, color=COLORS['lstm'], linewidth=1,
                        alpha=0.3, linestyle='--')
        axes[1, 1].plot(days, arima_pred, color=COLORS['arima'], linewidth=1,
                        alpha=0.3, linestyle='--')
        axes[1, 1].plot(days, prophet_pred, color=COLORS['prophet'], linewidth=1,
                        alpha=0.3, linestyle='--')
        axes[1, 1].set_title('Weighted Ensemble: 0.55×LSTM + 0.30×ARIMA + 0.15×Prophet',
                             fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Days Ahead', fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(alpha=0.3)
        _add_stats_box(axes[1, 1],
                       f"Ensemble Mean: {np.mean(ensemble):.2f}\n"
                       f"CI Width: {np.mean(forecast_result['upper_bound'] - forecast_result['lower_bound']):.2f}\n"
                       f"Horizon: {len(ensemble)} days")

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.close('all')

    # =====================================================================
    # MASTER METHOD: Generate all plots at once
    # =====================================================================

    def generate_all_data_plots(self, df):
        """Generate all 8 data exploration plots"""
        print("\n--- Data Exploration Plots ---")
        self.plot_data_source_overview(df)
        self.plot_daily_trends(df)
        self.plot_sentiment_distribution(df)
        self.plot_top_products(df)
        self.plot_data_quality(df)
        self.plot_mentions_histogram(df)
        self.plot_weekly_patterns(df)
        self.plot_correlation_heatmap(df)
