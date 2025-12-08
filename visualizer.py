"""
Visualization Module
Creates plots for dissertation and dashboard
"""

import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11


class Visualizer:
    """Create visualizations for analysis and reporting"""

    def __init__(self):
        self.plots_dir = PLOTS_DIR

    def plot_forecast_with_actual(self, product_df, forecast_result,
                                  train_days=120, save_name=None):
        """
        Plot forecast vs actual with confidence intervals

        Args:
            product_df: Full product DataFrame
            forecast_result: Dict from model.ensemble_forecast()
            train_days: Number of training days
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(16, 7))

        # Historical data
        dates = product_df['date'].values
        mentions = product_df['mentions'].values

        # Split point
        train_dates = dates[:train_days]
        test_dates = dates[train_days:train_days + len(forecast_result['forecast'])]

        # Plot historical
        ax.plot(train_dates, mentions[:train_days],
                'b-', linewidth=2.5, label='Historical Data', alpha=0.8)

        # Plot actual test data
        ax.plot(test_dates, mentions[train_days:train_days + len(test_dates)],
                'g-', linewidth=2.5, label='Actual (Test Period)', alpha=0.8)

        # Plot forecast
        ax.plot(test_dates, forecast_result['forecast'],
                'r--', linewidth=2.5, label='Ensemble Forecast', alpha=0.9)

        # Confidence interval
        ax.fill_between(test_dates,
                        forecast_result['lower_bound'],
                        forecast_result['upper_bound'],
                        alpha=0.25, color='red',
                        label='95% Confidence Interval')

        # Mark early warning zone (45-60 days before end of training)
        warning_date = train_dates[-45] if len(train_dates) >= 45 else train_dates[0]
        ax.axvline(warning_date, color='orange', linestyle=':',
                   linewidth=2.5, label='Early Warning Window', alpha=0.7)

        # Formatting
        product_name = product_df['product'].iloc[0] if 'product' in product_df.columns else 'Product'
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax.set_ylabel('Product Mentions', fontsize=13, fontweight='bold')
        ax.set_title(f'AI Trend Forecasting: {product_name}',
                     fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.plots_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")

        plt.show()

    def plot_model_comparison(self, results_df, save_name='model_comparison.png'):
        """
        Compare metrics across products

        Args:
            results_df: DataFrame from validator
            save_name: Filename to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # MAPE distribution
        axes[0, 0].hist(results_df['MAPE'], bins=15, color='steelblue',
                        edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(30, color='red', linestyle='--',
                           linewidth=2, label='Target: 30%')
        axes[0, 0].set_xlabel('MAPE (%)', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('MAPE Distribution Across Products',
                             fontsize=14, fontweight='bold')
        axes[0, 0].legend()

        # Accuracy by product
        top_10 = results_df.nsmallest(10, 'MAPE')
        axes[0, 1].barh(range(len(top_10)), top_10['Accuracy'],
                        color='green', alpha=0.7)
        axes[0, 1].set_yticks(range(len(top_10)))
        axes[0, 1].set_yticklabels([p[:20] for p in top_10['Product']], fontsize=10)
        axes[0, 1].set_xlabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Top 10 Products by Accuracy',
                             fontsize=14, fontweight='bold')
        axes[0, 1].axvline(70, color='red', linestyle='--', linewidth=2)

        # Peak timing error
        axes[1, 0].scatter(results_df['Actual_Peak_Day'],
                           results_df['Predicted_Peak_Day'],
                           alpha=0.6, s=100, color='purple')
        max_peak = max(results_df['Actual_Peak_Day'].max(),
                       results_df['Predicted_Peak_Day'].max())
        axes[1, 0].plot([0, max_peak], [0, max_peak], 'r--',
                        linewidth=2, label='Perfect Prediction')
        axes[1, 0].set_xlabel('Actual Peak Day', fontsize=12)
        axes[1, 0].set_ylabel('Predicted Peak Day', fontsize=12)
        axes[1, 0].set_title('Peak Timing Prediction',
                             fontsize=14, fontweight='bold')
        axes[1, 0].legend()

        # Early detection success
        success_rate = results_df['Early_Detection_Success'].mean() * 100
        categories = ['Success', 'Failed']
        values = [success_rate, 100 - success_rate]
        colors = ['green', 'red']
        axes[1, 1].pie(values, labels=categories, autopct='%1.1f%%',
                       colors=colors, startangle=90, textprops={'fontsize': 12})
        axes[1, 1].set_title('Early Detection Success Rate (45-60 Days)',
                             fontsize=14, fontweight='bold')

        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {save_path}")
        plt.show()

    def plot_trend_scores(self, df_scored, top_n=15, save_name='trend_scores.png'):
        """
        Plot trend scores for top products

        Args:
            df_scored: DataFrame with trend_score column
            top_n: Number of products to show
            save_name: Filename to save
        """
        # Get latest score for each product
        latest_scores = df_scored.groupby('product').apply(
            lambda x: x.nlargest(7, 'date')['trend_score'].mean()
        ).reset_index(name='avg_score')

        top_products = latest_scores.nlargest(top_n, 'avg_score')

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = ['red' if score > 70 else 'orange' if score > 50 else 'green'
                  for score in top_products['avg_score']]

        bars = ax.barh(range(len(top_products)), top_products['avg_score'],
                       color=colors, alpha=0.7, edgecolor='black')

        ax.set_yticks(range(len(top_products)))
        ax.set_yticklabels([p[:30] for p in top_products['product']], fontsize=10)
        ax.set_xlabel('Trend Score (0-100)', fontsize=13, fontweight='bold')
        ax.set_title('Top Trending Products - Real-Time Scores',
                     fontsize=15, fontweight='bold', pad=20)

        # Add threshold lines
        ax.axvline(60, color='red', linestyle='--', linewidth=2,
                   label='High Potential Threshold', alpha=0.7)
        ax.axvline(40, color='orange', linestyle='--', linewidth=2,
                   label='Medium Potential', alpha=0.7)

        ax.legend(loc='lower right', fontsize=11)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Trend scores plot saved to {save_path}")
        plt.show()

    def plot_component_breakdown(self, forecast_result, save_name='components.png'):
        """
        Plot individual model components

        Args:
            forecast_result: Dict from ensemble_forecast
            save_name: Filename to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        days = range(len(forecast_result['forecast']))

        # LSTM
        axes[0, 0].plot(days, forecast_result['components']['lstm'],
                        'b-', linewidth=2, label='LSTM Prediction')
        axes[0, 0].set_title('LSTM Model (50% weight)', fontsize=13, fontweight='bold')
        axes[0, 0].set_xlabel('Days Ahead')
        axes[0, 0].set_ylabel('Predicted Mentions')
        axes[0, 0].grid(alpha=0.3)

        # ARIMA
        axes[0, 1].plot(days, forecast_result['components']['arima'],
                        'g-', linewidth=2, label='ARIMA Prediction')
        axes[0, 1].set_title('ARIMA Model (30% weight)', fontsize=13, fontweight='bold')
        axes[0, 1].set_xlabel('Days Ahead')
        axes[0, 1].set_ylabel('Predicted Mentions')
        axes[0, 1].grid(alpha=0.3)

        # Prophet
        axes[1, 0].plot(days, forecast_result['components']['prophet'],
                        'orange', linewidth=2, label='Prophet Prediction')
        axes[1, 0].set_title('Prophet Model (20% weight)', fontsize=13, fontweight='bold')
        axes[1, 0].set_xlabel('Days Ahead')
        axes[1, 0].set_ylabel('Predicted Mentions')
        axes[1, 0].grid(alpha=0.3)

        # Ensemble
        axes[1, 1].plot(days, forecast_result['forecast'],
                        'r-', linewidth=2.5, label='Ensemble')
        axes[1, 1].fill_between(days, forecast_result['lower_bound'],
                                forecast_result['upper_bound'],
                                alpha=0.3, color='red')
        axes[1, 1].set_title('Final Ensemble Forecast', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Days Ahead')
        axes[1, 1].set_ylabel('Predicted Mentions')
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Component breakdown saved to {save_path}")
        plt.show()


# Test visualizer
if __name__ == "__main__":
    from data_loader import KaggleDataLoader
    from forecasting_model import HybridForecastingModel
    from trend_scorer import TrendScorer

    print("Testing Visualizer...")

    # Load data
    loader = KaggleDataLoader()
    df = loader.load_and_merge_all()

    # Get one product
    product = df['product'].unique()[0]
    product_df = df[df['product'] == product].copy()

    # Generate forecast
    model = HybridForecastingModel()
    train_df = product_df[:120]
    forecast = model.ensemble_forecast(train_df)

    # Create visualizations
    viz = Visualizer()

    print("\n1. Creating forecast plot...")
    viz.plot_forecast_with_actual(product_df, forecast,
                                  save_name=f'forecast_{product}.png')

    print("\n2. Creating component breakdown...")
    viz.plot_component_breakdown(forecast, save_name='model_components.png')

    # Trend scores
    print("\n3. Creating trend scores plot...")
    scorer = TrendScorer()
    df_scored = df.groupby('product', group_keys=False).apply(
        lambda x: scorer.calculate_trend_score(x)
    )
    viz.plot_trend_scores(df_scored, save_name='trend_leaderboard.png')