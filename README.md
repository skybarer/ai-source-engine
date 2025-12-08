ai-trend-source-engine/
│
├── data/                          # Kaggle datasets go here
│   ├── raw/
│   │   ├── amazon_sales.csv
│   │   ├── flipkart_products.csv
│   │   └── reddit_comments.csv
│   └── processed/
│       └── trend_data.csv
│
├── models/                        # Saved trained models
│   ├── lstm_model.h5
│   ├── scaler.pkl
│   └── results.json
│
├── notebooks/                     # Jupyter notebooks (optional)
│   └── exploratory_analysis.ipynb
│
├── src/                          # Source code
│   ├── __init__.py
│   ├── data_loader.py           # Load and preprocess Kaggle data
│   ├── trend_scorer.py          # Trend scoring algorithm
│   ├── forecasting_model.py     # LSTM + ARIMA + Prophet
│   ├── validator.py             # Metrics and backtesting
│   └── visualizer.py            # Create plots
│
├── outputs/                      # Results and plots
│   ├── plots/
│   │   ├── forecast_product_1.png
│   │   └── metrics_comparison.png
│   └── results/
│       └── validation_metrics.csv
│
├── requirements.txt              # Python dependencies
├── config.py                     # Configuration settings
├── main.py                       # Main execution script
└── README.md                     # Project documentation



pip install -r requirements.txt