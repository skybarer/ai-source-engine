"""
=============================================================================
VISUALIZER.PY - Step 5: Generate Dissertation Plots
=============================================================================

PURPOSE: Create publication-quality figures for the dissertation.

PLOTS GENERATED:
  1. Forecast vs Actual    → Shows how well the model predicts
  2. Model Comparison      → MAPE distribution, accuracy bars, peak detection
  3. Trend Leaderboard     → Top products ranked by trend score
  4. Component Breakdown   → Individual LSTM/ARIMA/Prophet contributions
=============================================================================
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves to file, no popup)
import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11


class Visualizer:
    """Create dissertation-quality visualizations"""

    def __init__(self):
        self.plots_dir = PLOTS_DIR

    def plot_forecast_with_actual(self, product_df, forecast_result,
                                  train_days=120, save_name=None):
        """
        Plot: Historical data + forecast + confidence interval.
        
        Shows:
          Blue line  = historical data (training period)
          Green line = actual test data (what really happened)
          Red dashed = our forecast (what we predicted)
          Red shaded = 95% confidence interval
          Orange line = early warning window marker
        """
        fig, ax = plt.subplots(figsize=(16, 7))

        dates = product_df['date'].values
        mentions = product_df['mentions'].values
        train_days = min(train_days, len(product_df) - 1)
        
        forecast_len = len(forecast_result['forecast'])
        available_test = len(product_df) - train_days
        actual_test_len = min(forecast_len, available_test)

        train_dates = dates[:train_days]
        test_dates = dates[train_days:train_days + forecast_len]

        # If not enough dates, extend with synthetic dates
        if len(test_dates) < forecast_len:
            import pandas as pd
            last = pd.Timestamp(dates[-1])
            test_dates = pd.date_range(
                start=last + pd.Timedelta(days=1),
                periods=forecast_len, freq='D'
            ).values

        # Historical data (blue)
        ax.plot(train_dates, mentions[:train_days],
                'b-', linewidth=2.5, label='Historical Data', alpha=0.8)

        # Actual test data (green)
        if actual_test_len > 0:
            ax.plot(test_dates[:actual_test_len],
                    mentions[train_days:train_days + actual_test_len],
                    'g-', linewidth=2.5, label='Actual (Test)', alpha=0.8)

        # Our forecast (red dashed)
        ax.plot(test_dates, forecast_result['forecast'],
                'r--', linewidth=2.5, label='Ensemble Forecast', alpha=0.9)

        # 95% confidence interval (red shaded)
        ax.fill_between(test_dates,
                        forecast_result['lower_bound'],
                        forecast_result['upper_bound'],
                        alpha=0.25, color='red', label='95% CI')

        # Early warning line
        warning_date = train_dates[-45] if len(train_dates) >= 45 else train_dates[0]
        ax.axvline(warning_date, color='orange', linestyle=':',
                   linewidth=2.5, label='Early Warning Window', alpha=0.7)

        name = product_df['product'].iloc[0] if 'product' in product_df.columns else 'Product'
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax.set_ylabel('Product Mentions', fontsize=13, fontweight='bold')
        ax.set_title(f'AI Trend Forecasting: {name}', fontsize=15, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_name:
            plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved {save_name}")
        plt.show()

    def plot_model_comparison(self, results_df, save_name='model_comparison.png'):
        """
        4-panel comparison plot:
          Top-left:     MAPE histogram (are predictions mostly good?)
          Top-right:    Accuracy bar chart (top 10 products)
          Bottom-left:  Peak timing scatter (predicted vs actual peak day)
          Bottom-right: Early detection pie chart (success rate)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # MAPE distribution
        axes[0, 0].hist(results_df['MAPE'], bins=15, color='steelblue',
                        edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(30, color='red', linestyle='--', linewidth=2, label='Target: 30%')
        axes[0, 0].set_xlabel('MAPE (%)')
        axes[0, 0].set_title('MAPE Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].legend()

        # Accuracy by product
        top10 = results_df.nsmallest(10, 'MAPE')
        axes[0, 1].barh(range(len(top10)), top10['Accuracy'], color='green', alpha=0.7)
        axes[0, 1].set_yticks(range(len(top10)))
        axes[0, 1].set_yticklabels([p[:20] for p in top10['Product']], fontsize=10)
        axes[0, 1].set_xlabel('Accuracy (%)')
        axes[0, 1].set_title('Top 10 by Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].axvline(70, color='red', linestyle='--', linewidth=2)

        # Peak timing
        axes[1, 0].scatter(results_df['Actual_Peak_Day'],
                           results_df['Predicted_Peak_Day'],
                           alpha=0.6, s=100, color='purple')
        m = max(results_df['Actual_Peak_Day'].max(), results_df['Predicted_Peak_Day'].max())
        axes[1, 0].plot([0, m], [0, m], 'r--', linewidth=2, label='Perfect')
        axes[1, 0].set_xlabel('Actual Peak Day')
        axes[1, 0].set_ylabel('Predicted Peak Day')
        axes[1, 0].set_title('Peak Timing Prediction', fontsize=14, fontweight='bold')
        axes[1, 0].legend()

        # Early detection success
        sr = results_df['Early_Detection_Success'].mean() * 100
        axes[1, 1].pie([sr, 100 - sr], labels=['Success', 'Failed'],
                       autopct='%1.1f%%', colors=['green', 'red'],
                       startangle=90, textprops={'fontsize': 12})
        axes[1, 1].set_title('Early Detection (45-60 Days)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.show()

    def plot_trend_scores(self, df_scored, top_n=15, save_name='trend_scores.png'):
        """Horizontal bar chart of top trending products by score"""
        latest = df_scored.groupby('product').apply(
            lambda x: x.nlargest(7, 'date')['trend_score'].mean()
        ).reset_index(name='avg_score')

        top = latest.nlargest(top_n, 'avg_score')

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['red' if s > 70 else 'orange' if s > 50 else 'green'
                  for s in top['avg_score']]

        ax.barh(range(len(top)), top['avg_score'], color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([p[:30] for p in top['product']], fontsize=10)
        ax.set_xlabel('Trend Score (0-100)', fontsize=13, fontweight='bold')
        ax.set_title('Top Trending Products', fontsize=15, fontweight='bold')
        ax.axvline(60, color='red', linestyle='--', linewidth=2, label='High Potential', alpha=0.7)
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.show()

    def plot_component_breakdown(self, forecast_result, save_name='components.png'):
        """Show each model's individual prediction vs the ensemble"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        days = range(len(forecast_result['forecast']))

        # LSTM
        axes[0, 0].plot(days, forecast_result['components']['lstm'], 'b-', linewidth=2)
        axes[0, 0].set_title('LSTM (55% weight)', fontsize=13, fontweight='bold')
        axes[0, 0].set_ylabel('Predicted Mentions')
        axes[0, 0].grid(alpha=0.3)

        # ARIMA
        axes[0, 1].plot(days, forecast_result['components']['arima'], 'g-', linewidth=2)
        axes[0, 1].set_title('ARIMA (30% weight)', fontsize=13, fontweight='bold')
        axes[0, 1].grid(alpha=0.3)

        # Prophet
        axes[1, 0].plot(days, forecast_result['components']['prophet'], color='orange', linewidth=2)
        axes[1, 0].set_title('Prophet (15% weight)', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Days Ahead')
        axes[1, 0].grid(alpha=0.3)

        # Ensemble
        axes[1, 1].plot(days, forecast_result['forecast'], 'r-', linewidth=2.5)
        axes[1, 1].fill_between(days, forecast_result['lower_bound'],
                                forecast_result['upper_bound'], alpha=0.3, color='red')
        axes[1, 1].set_title('Ensemble (Final)', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Days Ahead')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved {save_name}")
        plt.show()
