# EVERY CONCEPT IN THIS PROJECT + EVERY POSSIBLE INSTRUCTOR QUESTION

> **Student:** INKOLLU AKASHDHAR (2023AC05051) | **M.Tech Dissertation, BITS Pilani (WILP)**  
> **Project:** AI Trend-to-Source Engine — Predicting E-Commerce Product Trends 45-60 Days in Advance  
> **Key Result:** MAPE ~12.80% → Accuracy ~87.20% (target was >70%)

---

## TABLE OF CONTENTS

1. [Machine Learning Fundamentals](#1-machine-learning-fundamentals)
2. [Deep Learning — LSTM](#2-deep-learning--lstm)
3. [Statistical Models — ARIMA](#3-statistical-models--arima)
4. [Time Series — Facebook Prophet](#4-time-series--facebook-prophet)
5. [Ensemble Methods](#5-ensemble-methods)
6. [Data Preprocessing & Feature Engineering](#6-data-preprocessing--feature-engineering)
7. [Sentiment Analysis](#7-sentiment-analysis)
8. [Trend Scoring Algorithm](#8-trend-scoring-algorithm)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Training Techniques & Regularization](#10-training-techniques--regularization)
11. [Data Augmentation](#11-data-augmentation)
12. [Time Series Concepts](#12-time-series-concepts)
13. [Software Engineering & Architecture](#13-software-engineering--architecture)
14. [Data Sources & Dataset Knowledge](#14-data-sources--dataset-knowledge)
15. [Visualization & Interpretation](#15-visualization--interpretation)
16. [Failure Cases & Limitations](#16-failure-cases--limitations)
17. [Academic Context & Literature](#17-academic-context--literature)
18. [Business Value & Real-World Application](#18-business-value--real-world-application)
19. [Mathematical Foundations](#19-mathematical-foundations)
20. [Quick-Fire 50 Questions](#20-quick-fire-50-questions)

---

## 1. MACHINE LEARNING FUNDAMENTALS

### Concept: Supervised Learning
- **What:** Model learns from labeled data (input → known output)
- **In our project:** We have historical mention counts (input) and future mention counts (target). Model learns to map past → future.
- **Q: Is this supervised or unsupervised?**
  - **A:** Supervised regression. Input = last 20 days of mentions. Output = next 15 days of mentions. Labels are the actual future values.

### Concept: Regression vs Classification
- **What:** Regression predicts continuous values; classification predicts categories
- **In our project:** This is a **regression** problem — we predict continuous mention counts (e.g., 12.5, 45.3), not categories.
- **Q: Why not classification (trending/not-trending)?**
  - **A:** Regression gives us exact predicted values which we can use to compute MAPE and detect peaks. Classification would only give yes/no, losing information about magnitude and timing.

### Concept: Train/Test Split
- **What:** Divide data into training set (model learns from) and test set (model is evaluated on, NEVER seen during training)
- **In our project:** First 200 days = training, last 15 days = test
- **Q: Why not random split? Why sequential?**
  - **A:** Time series data is ordered. Random splitting would cause **data leakage** — the model would "see" future data during training. We always split chronologically: past → train, future → test.
- **Q: Why 200 train and 15 test?**
  - **A:** 200 days gives ~165 sliding window sequences for LSTM (200 - 20 lookback - 15 horizon = 165). 15 test days matches the forecast horizon.

### Concept: Overfitting vs Underfitting
- **Overfitting:** Model memorizes training data, fails on unseen data (high train accuracy, low test accuracy)
- **Underfitting:** Model too simple to capture patterns (low on both)
- **In our project:** We prevent overfitting with: Dropout (15%), Early Stopping (patience=10), small batch size (8)
- **Q: How do you know you're not overfitting?**
  - **A:** Our test accuracy (~87%) is comparable to training accuracy. If overfitting, test accuracy would be much lower. Also, early stopping halts training when validation loss stops improving.

### Concept: Bias-Variance Tradeoff
- **Bias:** Error from overly simple models (underfitting)
- **Variance:** Error from overly complex models (overfitting)
- **In our project:** Ensemble reduces variance (combining 3 models smooths out individual errors). ARIMA has high bias (simple linear) but low variance. LSTM has low bias but higher variance. Combining them balances both.
- **Q: How does ensemble help with bias-variance?**
  - **A:** Each model has different bias/variance profile. LSTM = low bias, high variance. ARIMA = higher bias, lower variance. Combining them: ensemble bias ≈ weighted average bias, but variance is reduced because errors are uncorrelated.

### Concept: Hyperparameters vs Parameters
- **Parameters:** Learned by the model during training (LSTM weights & biases)
- **Hyperparameters:** Set by us before training (learning rate, layer sizes, dropout rate)
- **In our project:**
  - Parameters: ~500K+ LSTM weights (learned via Adam optimizer)
  - Hyperparameters: learning_rate=0.0008, lstm_units=[256,128,64], dropout=0.15, lookback=20, horizon=15, epochs=200, batch_size=8
- **Q: How did you choose hyperparameters?**
  - **A:** Grid search on key ones:
    - lookback: tested 10/20/30 → 20 best MAPE
    - learning_rate: tested 0.001/0.0008/0.0005 → 0.0008 best
    - dropout: tested 0.10/0.15/0.20 → 0.15 best
    - ARIMA order: tested (1,1,1)/(2,1,2)/(3,1,3) → (2,1,2) lowest AIC

### Concept: Feature Engineering
- **What:** Creating new input features from raw data to help the model
- **In our project:**
  - Raw: product_name, rating → Engineered: mentions (count), sentiment (rating/5.0), trend_score (4-factor)
  - 7-day moving average (smoothing)
  - Growth rate (% change of 7-day MA)
  - Saturation index (1 - current/max)
  - Acceleration (diff of growth rate)

### Concept: Cross-Validation
- **What:** Multiple train/test splits to get robust performance estimate
- **In our project:** We don't use k-fold CV because time series requires chronological splitting. Instead we use **expanding window** — train on first N days, test on next 15, then expand N.
- **Q: Why not k-fold cross-validation?**
  - **A:** k-fold randomly shuffles data, which violates temporal order. For time series, we use sequential validation (also called walk-forward validation).

---

## 2. DEEP LEARNING — LSTM

### Concept: Neural Network Basics
- **Neuron:** Receives inputs, multiplies by weights, adds bias, passes through activation function
- **Layer:** Collection of neurons processing together
- **Forward pass:** Data flows input → hidden layers → output
- **Backpropagation:** Error is propagated backward to update weights
- **Q: How does a neural network learn?**
  - **A:** Forward pass generates prediction → Loss function measures error → Backpropagation computes gradients (how much each weight contributed to error) → Optimizer (Adam) updates weights to reduce error → Repeat for many epochs.

### Concept: RNN (Recurrent Neural Network)
- **What:** Neural network with loops — output at time t feeds back as input at time t+1
- **Problem:** Vanilla RNNs suffer from **vanishing gradient** — when sequences are long (>10 steps), gradients during backpropagation shrink to near-zero, so the network can't learn long-term dependencies
- **Q: Why not use a simple RNN?**
  - **A:** Vanishing gradient problem. For 20-day lookback, a simple RNN would "forget" patterns from day 1-10 by the time it reaches day 20. LSTM solves this.

### Concept: LSTM (Long Short-Term Memory)
- **What:** Special RNN with gates that control information flow
- **Three gates:**
  1. **Forget Gate:** Decides what to throw away from cell state. σ(W_f · [h_{t-1}, x_t] + b_f)
  2. **Input Gate:** Decides what new info to store. σ(W_i · [h_{t-1}, x_t] + b_i)
  3. **Output Gate:** Decides what to output. σ(W_o · [h_{t-1}, x_t] + b_o)
- **Cell State:** The "memory highway" — allows information to flow unchanged across many time steps
- **In our project:** 3-layer LSTM: 256 → 128 → 64 units
- **Q: Explain the gates in LSTM.**
  - **A:** Think of cell state as a conveyor belt of memory:
    - **Forget gate:** "Should I forget sales dip from 2 weeks ago?" (yes if it was a one-time event)
    - **Input gate:** "Should I remember this new spike in mentions?" (yes if it's significant)
    - **Output gate:** "What should I predict today based on what I remember?" (filtered memory)
- **Q: Why 3 layers? Why 256→128→64?**
  - **A:** Pyramid/funnel architecture. Layer 1 (256): captures raw temporal patterns from 20-day window. Layer 2 (128): abstracts higher-level features (trends, cycles). Layer 3 (64): refines into compact representation for prediction. Going wider (512+) overfits on our small data. Going shallower (1 layer) underfit.
- **Q: What does "hidden units" mean?**
  - **A:** Number of neurons in an LSTM layer. 256 units = the LSTM layer has 256 independent memory cells, each tracking different aspects of the time series pattern.

### Concept: Batch Processing
- **What:** Instead of updating weights after every sample, update after a batch of samples
- **In our project:** batch_size = 8
- **Q: Why batch size 8?**
  - **A:** Our dataset has ~165 training sequences. Batch size 8 → ~20 weight updates per epoch. Small batches = more noisy but more regularized gradient updates. Large batches (32+) would mean only ~5 updates per epoch, too few for convergence. 8 is standard for small datasets.

### Concept: Sliding Window
- **What:** Create training sequences by sliding a fixed-size window across the time series
- **In our project:**
  ```
  Day 1-20  → predict Day 21-35    (Sequence 1)
  Day 2-21  → predict Day 22-36    (Sequence 2)
  Day 3-22  → predict Day 23-37    (Sequence 3)
  ...
  Day 166-185 → predict Day 186-200 (Sequence 165)
  ```
- **Q: How do you create training sequences for LSTM?**
  - **A:** Sliding window. Window size = lookback(20). We slide by 1 day. Each window is one training sample: input = 20 days, target = next 15 days. From 200 days of data, we get 200 - 20 - 15 = 165 sequences.

### Concept: Dense/Fully Connected Layer
- **What:** Every neuron connected to every neuron in next layer
- **In our project:** After 3 LSTM layers, we have Dense(64, ReLU) → Dense(15) to map LSTM's hidden state to 15 forecast values
- **Q: Why Dense layers after LSTM?**
  - **A:** LSTM output is a 64-dimensional hidden state (abstract representation). Dense layers transform this abstract representation into the actual 15-day forecast values.

### Concept: Activation Functions
- **ReLU (Rectified Linear Unit):** f(x) = max(0, x). Used between Dense layers.
  - **Why ReLU?** Fast, avoids vanishing gradient, works well in practice
  - **Why not sigmoid?** Sigmoid squishes to (0,1), causes vanishing grad in deep networks
- **No activation on output layer:** Because we need raw prediction values (regression), not probabilities
- **Q: Why ReLU and not sigmoid or tanh?**
  - **A:** ReLU is computationally faster, doesn't suffer from vanishing gradient in deep networks. Sigmoid/tanh squish values, ReLU passes positive values unchanged. Standard for hidden layers in modern deep learning.

---

## 3. STATISTICAL MODELS — ARIMA

### Concept: ARIMA (AutoRegressive Integrated Moving Average)
- **Full name breakdown:**
  - **AR (AutoRegressive):** Current value depends on past values
  - **I (Integrated):** Differencing to make data stationary
  - **MA (Moving Average):** Current value depends on past forecast errors
- **In our project:** ARIMA(2, 1, 2)
  - p=2: Uses last 2 values to predict
  - d=1: One round of differencing
  - q=2: Uses last 2 forecast errors

### Concept: ARIMA Parameters (p, d, q)
- **p (AR order = 2):** "Today's value depends on yesterday's and the day-before-yesterday's values"
  - Formula: y_t = c + φ₁·y_{t-1} + φ₂·y_{t-2} + ε_t
  - **Q: What does p=2 mean practically?**
    - **A:** The model looks back 2 time steps. If mentions were 50 yesterday and 45 the day before, it uses both values (with learned weights φ₁, φ₂) to predict today.

- **d (Differencing order = 1):** "Subtract each value from its previous value"
  - Original: [100, 105, 102, 108] → Differenced: [5, -3, 6]
  - **Q: Why d=1?**
    - **A:** To make the series **stationary** (constant mean/variance). Our mention data has a trend (increasing/decreasing over time). Differencing removes the trend, leaving only fluctuations. ARIMA requires stationarity.

- **q (MA order = 2):** "Adjust prediction based on last 2 forecast errors"
  - Formula: y_t = μ + ε_t + θ₁·ε_{t-1} + θ₂·ε_{t-2}
  - **Q: What does q=2 mean?**
    - **A:** If the model was off by +5 yesterday and -3 two days ago, it adjusts today's prediction using these error terms. This is a self-correcting mechanism.

### Concept: Stationarity
- **What:** A time series is stationary if its mean, variance, and autocorrelation don't change over time
- **Why needed:** ARIMA mathematically assumes stationarity. Non-stationary data leads to spurious results.
- **How we achieve it:** d=1 (first-order differencing)
- **Q: How do you test for stationarity?**
  - **A:** Augmented Dickey-Fuller (ADF) test. Null hypothesis: series is non-stationary. If p-value < 0.05, reject null → series is stationary. After d=1 differencing, our data passes ADF test.

### Concept: AIC (Akaike Information Criterion)
- **What:** Measures model quality: balances fit vs. complexity. Lower AIC = better.
- **Formula:** AIC = 2k - 2ln(L), where k = number of parameters, L = likelihood
- **In our project:** Tested (1,1,1), (2,1,2), (3,1,3). ARIMA(2,1,2) had lowest AIC.
- **Q: How did you choose (2,1,2)?**
  - **A:** AIC criterion. (1,1,1) underfits (high AIC), (3,1,3) overfits (parameters > data supports). (2,1,2) is the sweet spot — lowest AIC with good generalization.

### Concept: Autocorrelation (ACF) and Partial Autocorrelation (PACF)
- **ACF:** Correlation between a time series and its lagged version at various lags
- **PACF:** Direct correlation at a specific lag, removing effects of intermediate lags
- **Used for:** ACF → determines q (MA order), PACF → determines p (AR order)
- **Q: How do you determine ARIMA order from data?**
  - **A:** Plot PACF → significant lags until cutoff = p. Plot ACF → significant lags until cutoff = q. Our PACF showed significance at lag 1 and 2 (→ p=2), ACF showed significance at lag 1 and 2 (→ q=2). d=1 determined by ADF test.

---

## 4. TIME SERIES — FACEBOOK PROPHET

### Concept: Prophet (Time Series Decomposition)
- **What:** Facebook's open-source tool for business time series forecasting
- **Decomposition:** y(t) = g(t) + s(t) + h(t) + ε(t)
  - g(t) = growth/trend (piecewise linear or logistic)
  - s(t) = seasonality (weekly, yearly patterns)
  - h(t) = holiday effects
  - ε(t) = noise
- **In our project:** Weekly seasonality enabled, multiplicative mode
- **Q: What makes Prophet different from ARIMA?**
  - **A:** Prophet is **additive/multiplicative decomposition** (trend + seasonality + holidays). ARIMA is **autoregressive** (past values + errors). Prophet handles missing data and outliers better. ARIMA is more mathematically rigorous for linear patterns.

### Concept: Additive vs Multiplicative Seasonality
- **Additive:** Seasonal effect is constant. If base is 100, weekend adds +20 → 120. If base grows to 200, weekend still adds +20 → 220.
- **Multiplicative:** Seasonal effect scales with level. If base is 100, weekend is ×1.2 → 120. If base grows to 200, weekend is ×1.2 → 240.
- **In our project:** **Multiplicative**
- **Q: Why multiplicative?**
  - **A:** E-commerce seasonal effects SCALE with product popularity. A trending product with 1000 mentions has bigger weekend spikes than a niche product with 10 mentions. The percentage increase is what's constant, not the absolute increase.

### Concept: Weekly Seasonality
- **What:** Recurring 7-day pattern (e.g., more shopping on weekends, pay days)
- **In our project:** Prophet captures this automatically with Fourier series
- **Q: Why weekly but not daily or yearly?**
  - **A:** Daily: our data is already daily-aggregated, so no intra-day patterns. Yearly: we only have 180-360 days of data, not enough for yearly seasonality (need 2+ years). Weekly: clear e-commerce pattern (weekend browsing, Monday purchasing).

---

## 5. ENSEMBLE METHODS

### Concept: Ensemble Learning
- **What:** Combining multiple models to get better predictions than any single model
- **Types:** Bagging, Boosting, Stacking, Weighted Average
- **In our project:** **Weighted average ensemble**
  - Formula: forecast = 0.55 × LSTM + 0.30 × ARIMA + 0.15 × Prophet
- **Q: Why not just use the best single model?**
  - **A:** M4 Competition (Makridakis et al., 2020) showed ensemble methods consistently outperform single models by 10-15%. Each model captures different patterns — LSTM gets nonlinear, ARIMA gets linear autoregressive, Prophet gets seasonality. Errors are uncorrelated, so combining reduces overall variance.

### Concept: Weighted Average vs Simple Average
- **Simple average:** Each model gets equal weight (33/33/33)
- **Weighted average:** Different weights based on model reliability
- **In our project:** 55/30/15 (not equal)
- **Q: Why not equal weights?**
  - **A:** Tested both. Equal weights: ~72% accuracy. Current weights: ~87% accuracy. LSTM is most capable (captures nonlinear patterns), so gets highest weight. Prophet is least reliable on short series, so gets lowest.
- **Q: How did you determine 55/30/15?**
  - **A:** Iterative testing:
    - (33/33/33) → 72%
    - (50/30/20) → 78%
    - (55/30/15) → 87%
    - (70/20/10) → 82% (too LSTM-dependent)
  
    55/30/15 was the empirical optimum.

### Concept: Model Diversity
- **What:** For ensemble to work, models must make **different types of errors**
- **In our project:**
  - LSTM: overpredicts peaks, good on trends
  - ARIMA: conservative, good on stable periods
  - Prophet: captures periodic patterns but noisy on non-seasonal data
  - These errors are uncorrelated → averaging reduces them

### Concept: Confidence Intervals
- **What:** Range within which the true value falls with certain probability
- **In our project:** 95% CI = forecast ± 1.96 × std(LSTM predictions)
- **Q: How do you calculate confidence intervals?**
  - **A:** We use the standard deviation of LSTM's forecast as a proxy for uncertainty, multiplied by 1.96 (z-score for 95% confidence). CI = ensemble_forecast ± 1.96 × 1.5 × std(lstm_predictions). The 1.5 factor accounts for additional variance from ARIMA and Prophet.

---

## 6. DATA PREPROCESSING & FEATURE ENGINEERING

### Concept: Data Aggregation
- **What:** Combining multiple rows into summary statistics
- **In our project:** Raw data = 1 row per review. Aggregated = 1 row per (date, product) with COUNT(reviews) → mentions, MEAN(sentiment) → sentiment
- **Q: Why aggregate?**
  - **A:** Individual reviews are too noisy. Aggregating by day gives us a time series of review activity that the model can learn from.

### Concept: MinMaxScaler (Normalization)
- **Formula:** x_scaled = (x - x_min) / (x_max - x_min)
- **Range:** Maps values to [0, 1]
- **In our project:** Applied to mention counts before LSTM input
- **Q: Why scale to [0,1]?**
  - **A:** LSTM (and neural networks in general) learn best with small input values. Raw mentions (0-500+) would cause large gradients and slow/unstable training. Scaling to [0,1] ensures all features contribute equally and gradients are manageable.
- **Q: Do you inverse-transform predictions?**
  - **A:** Yes. After LSTM predicts in [0,1] range, we use scaler.inverse_transform() to get back to original mention counts.
- **Q: Why MinMax and not StandardScaler?**
  - **A:** MinMax preserves zero values (important — zero mentions means no reviews). StandardScaler can produce negative values, which don't make sense for mention counts.

### Concept: Missing Data Handling
- **What:** Dealing with NaN/null values
- **In our project:** 
  - fillna(0) for mentions (no reviews = 0 mentions)
  - fillna(0.5) for sentiment (unknown sentiment = neutral)
  - dropna() for rows with no product name or rating
- **Q: How do you handle missing data?**
  - **A:** Reviews with no rating are dropped (can't compute sentiment). Missing mention days get 0 (no review that day). Missing sentiment defaults to 0.5 (neutral — conservative assumption).

### Concept: Moving Average (Smoothing)
- **What:** Replace each value with the average of surrounding values
- **In our project:** 7-day backward-looking moving average
- **Q: Why 7-day? Why backward?**
  - **A:** 7-day: matches weekly business cycles (captures one full week of pattern). Backward-looking: only uses past data (day t = average of days t-6 to t). Using future data would be **data leakage** — the model would "know" the future during training.
- **Noise reduction:** 84% reduction in standard deviation after smoothing

---

## 7. SENTIMENT ANALYSIS

### Concept: Sentiment Analysis
- **What:** Determining whether text/opinion is positive, negative, or neutral
- **In our project:** `sentiment = star_rating / 5.0`
  - Rating 5 → 1.0 (very positive)
  - Rating 3 → 0.6 (neutral)
  - Rating 1 → 0.2 (very negative)

### Concept: Rating-Based vs NLP-Based Sentiment
- **Rating-based (our approach):**
  - Pro: 95%+ accuracy, directly from customer
  - Pro: Fast computation, no model needed
  - Con: Only 1-5 granularity, no aspect-level sentiment
- **NLP-based (alternative):**
  - Pro: Fine-grained, aspect-level (e.g., "battery good, camera bad")
  - Con: ~70% accuracy on short reviews
  - Con: Needs pre-trained model (BERT/VADER)

- **Q: Why not use BERT or VADER for sentiment?**
  - **A:** Star ratings are more reliable. A customer who writes "good product" but gives 2 stars is actually dissatisfied — the star rating captures true intent. VADER would classify "good product" as positive, missing the nuance. For our forecasting purpose, the overall sentiment signal (positive/negative) is what matters, and star ratings give that with >95% reliability.

- **Q: Is rating/5.0 really sentiment analysis?**
  - **A:** Yes — it's **explicit sentiment**. The customer explicitly expressed their satisfaction level (1-5 stars). This is a valid and widely-used form of sentiment analysis in e-commerce literature (Hu & Liu, 2004; Pang & Lee, 2008). More sophisticated NLP adds marginal value for our forecasting use case.

---

## 8. TREND SCORING ALGORITHM

### Concept: Multi-Factor Scoring
- **What:** Combining multiple independent indicators into a single score
- **In our project:** 4-factor trend score (0-100):
  ```
  trend_score = growth(40%) + sentiment(20%) + saturation(20%) + profit(20%)
  ```

### Factor 1: Growth Velocity (40%)
- **Formula:** 7-day MA of mentions → pct_change over 7 days → clipped to [0, 40]
- **What it measures:** How fast are mentions increasing?
- **Why 40%:** Strongest single predictor of future peaks (r=0.65 correlation)
- **Q: What correlation did you measure?**
  - **A:** Pearson r=0.65 between growth velocity and whether a product peaked within 30 days. This was the highest among all 4 factors.

### Factor 2: Sentiment Polarity (20%)
- **Formula:** mean(sentiment) × 20
- **What it measures:** Are reviews positive? (rules out controversy-driven spikes)
- **Why 20%:** Validates trend quality, not just quantity

### Factor 3: Saturation Index (20%)
- **Formula:** (1 - current_mentions / cumulative_max) × 20
- **What it measures:** Has the product already peaked?
- **Why 20%:** Penalizes products past their prime
- **Q: What happens when current = max?**
  - **A:** Saturation = 0 → product is at its all-time high → future growth potential is low. If current << max, saturation is high → product has crashed from peak → room to grow again.

### Factor 4: Profit Potential (20%)
- **Formula:** diff(growth_rate) × 20 (second derivative)
- **What it measures:** Is growth accelerating or decelerating?
- **Why 20%:** Positive acceleration = trend building momentum
- **Q: What's a second derivative in this context?**
  - **A:** First derivative = growth rate (is it going up?). Second derivative = acceleration (is it going up FASTER?). Positive acceleration means the trend is strengthening — the most profitable time to source a product.

### Concept: Early Warning System
- **Threshold:** trend_score > 60 AND velocity > 5
- **What it means:** Product is likely to peak within 45-60 days
- **Q: Why 60 as threshold?**
  - **A:** Tested 50/60/70. At 50: too many false positives (products that don't actually peak). At 70: too few alerts (misses real trends). 60 was the best precision-recall balance.

---

## 9. EVALUATION METRICS

### Concept: MAPE (Mean Absolute Percentage Error)
- **Formula:** MAPE = (1/n) × Σ |actual_i - predicted_i| / |actual_i| × 100%
- **Our value:** ~12.80%
- **Target:** <30% (= accuracy >70%)
- **Q: Why MAPE as primary metric?**
  - **A:** Scale-independent — works for products with 10 or 10,000 mentions. Standard in time series literature (M4 Competition, Makridakis et al., 2018). Interpretation: "On average, our predictions are off by 12.80%"
- **Q: What are MAPE benchmarks?**
  - **A:** <10% = excellent, 10-20% = good, 20-30% = fair, >30% = poor. Our 12.80% = "good"
- **Limitation:** MAPE is undefined when actual = 0. We handle this with smoothing (ensures no zero values).

### Concept: MAE (Mean Absolute Error)
- **Formula:** MAE = (1/n) × Σ |actual_i - predicted_i|
- **What it measures:** Average absolute error in original units (mentions)
- **Advantage:** Easy to interpret ("predictions are off by X mentions on average")
- **Disadvantage:** Not scale-independent — MAE of 10 means different things for a product with 20 mentions vs 2000

### Concept: RMSE (Root Mean Squared Error)
- **Formula:** RMSE = √[(1/n) × Σ (actual_i - predicted_i)²]
- **What it measures:** Similar to MAE but penalizes large errors more (due to squaring)
- **Q: When is RMSE >> MAE?**
  - **A:** When there are occasional large outlier errors. RMSE squares errors, so one prediction that's off by 50 contributes 2500 to RMSE but only 50 to MAE. If RMSE ≈ MAE, errors are uniform.

### Concept: Accuracy (= 100 - MAPE)
- **Our value:** ~87.20%
- **Q: How do you define accuracy for regression?**
  - **A:** Accuracy = 100 - MAPE. If MAPE is 12.80%, accuracy is 87.20%. This means "on average, our predictions capture 87.20% of the actual value."

### Concept: Peak Detection Accuracy
- **What:** |day_of_actual_peak - day_of_predicted_peak|
- **Our target:** ±7 days
- **Q: Why ±7 days tolerance?**
  - **A:** Business requirement. A seller needs to know WHEN to stock up. Being off by ±7 days is acceptable — they can adjust inventory. Being off by ±30 days negates the early warning value.

### Concept: Direction Accuracy
- **What:** Did we predict the correct direction (up/down) of the trend?
- **Formula:** % of days where sign(actual_change) = sign(predicted_change)
- **Q: Why track direction in addition to MAPE?**
  - **A:** A seller cares whether the product is trending UP or DOWN. Even if magnitude is slightly off, correct direction is valuable for sourcing decisions.

---

## 10. TRAINING TECHNIQUES & REGULARIZATION

### Concept: Dropout
- **What:** Randomly set a fraction of neurons to zero during training
- **In our project:** 15% dropout after each LSTM layer and before output
- **Q: How does dropout prevent overfitting?**
  - **A:** Forces the network to not rely on any single neuron. Each neuron must be independently useful because it might be "dropped out." This creates redundancy and generalization. At inference time, all neurons are active, but weights are scaled.
- **Q: Why 15%?**
  - **A:** Tested 10% (still overfitting), 15% (optimal), 20% (underfitting). 15% is the sweet spot for our dataset size.

### Concept: Early Stopping
- **What:** Stop training when validation loss stops improving
- **In our project:** Patience = 10 epochs. If loss doesn't improve for 10 consecutive epochs, stop.
- **Q: Why not just train for all 200 epochs?**
  - **A:** After the optimal point, the model starts overfitting (memorizing training noise). Early stopping saves the best model state and reverts to it. Typically stops around epoch 50-100.
- **Q: How does early stopping work in your code?**
  - **A:** We track best_loss. Each epoch: if loss < best_loss, save model state and reset patience counter. Else increment patience. When patience hits 10, break training loop and restore best saved state.

### Concept: Adam Optimizer
- **What:** Adaptive Moment Estimation — adjusts learning rate per parameter
- **Combines:** Momentum (past gradients) + RMSprop (adaptive per-parameter rates)
- **In our project:** lr = 0.0008
- **Q: Why Adam and not SGD?**
  - **A:** Adam adapts learning rate for each parameter individually. SGD uses one global learning rate. Adam converges faster on small, noisy datasets. It's the de facto standard for deep learning (Kingma & Ba, 2015).
- **Q: Why learning rate 0.0008?**
  - **A:** Slightly below default 0.001. Smaller LR = finer convergence (smaller weight updates = more precise). On small data, aggressive updates (high LR) cause oscillation. 0.0008 was empirically best.

### Concept: MSE Loss Function
- **Formula:** MSE = (1/n) × Σ (actual_i - predicted_i)²
- **Why:** Standard for regression. Squaring penalizes large errors more than small ones.
- **Q: Why MSE not MAE as loss?**
  - **A:** MSE is smooth and differentiable everywhere (needed for gradient descent). MAE has a kink at 0 which can cause optimization issues. MSE also penalizes outliers more, which helps LSTM learn to avoid big errors.

### Concept: Gradient Descent
- **What:** Iteratively adjusting model weights in the direction that reduces loss
- **Formula:** w_new = w_old - learning_rate × ∂loss/∂w
- **In our project:** Adam optimizer performs this automatically
- **Q: What is a gradient?**
  - **A:** The gradient is the derivative of the loss function with respect to each weight. It tells us "if I increase this weight by a tiny amount, does the loss go up or down?" We move in the direction that decreases loss.

### Concept: Vanishing/Exploding Gradients
- **Vanishing:** Gradients become extremely small in deep networks → early layers don't learn
- **Exploding:** Gradients become extremely large → weights diverge
- **In our project:** LSTM gates prevent vanishing gradients. Dropout and small learning rate prevent exploding.
- **Q: How does LSTM prevent vanishing gradients?**
  - **A:** The cell state (memory highway) allows gradients to flow unchanged across time steps via the forget gate. If the forget gate is close to 1, the gradient passes through undiminished. This is the key innovation of LSTM over vanilla RNN.

---

## 11. DATA AUGMENTATION

### Concept: Data Augmentation
- **What:** Artificially increasing training data by creating modified versions of existing data or generating synthetic data
- **In our project:** Two techniques:
  1. **Smoothing** (7-day MA): De-noises existing data
  2. **Synthetic Generation**: 30 new products with realistic lifecycles

### Concept: Synthetic Data Generation
- **What:** Algorithmically creating data that mimics real-world patterns
- **In our project:** Each synthetic product has 3 lifecycle phases:
  1. **Pre-viral** (Day 0 to ~Day 50): Low, steady mentions (base level)
  2. **Growth** (Day ~50 to ~Day 120): Exponential increase (product going viral)
  3. **Decay** (Day ~120+): Exponential decrease (trend dying)
  - Plus weekly seasonality (sine wave with 7-day period)
- **Q: Why generate synthetic data?**
  - **A:** Real Kaggle data has no temporal structure (dates are synthetic anyway). Synthetic products have REAL lifecycle patterns (growth → peak → decay). This gives LSTM training sequences with patterns it can actually learn from. Data augmentation is standard ML practice (Shorten & Khoshgoftaar, 2019).
- **Q: How many synthetic products?**
  - **A:** 30 products × 360 days = 10,800 records. Added to ~21K real records → ~32K total. This roughly doubles the training sequences available to LSTM.

### Concept: Product Lifecycle Modeling
- **Growth curve:** mentions = base × (1 + (peak_mult-1) × progress^1.5)
  - progress^1.5: slightly sub-linear growth (realistic — not all products grow linearly)
- **Decay curve:** mentions = peak × e^(-decay_rate × days_past_peak)
  - Exponential decay: standard model for product/trend decline
- **Q: Why exponential growth and decay?**
  - **A:** Empirically observed in viral content (Crane & Sornette, 2008). Product interest follows: slow buildup → rapid growth → peak → exponential decay. This S-curve/bell-curve is standard in marketing/product lifecycle literature.

---

## 12. TIME SERIES CONCEPTS

### Concept: Time Series
- **What:** Sequence of observations ordered in time
- **In our project:** Daily mentions per product over 180-360 days
- **Properties:** Trend, Seasonality, Noise, Stationarity

### Concept: Trend
- **What:** Long-term increase or decrease in data
- **In our project:** Some products show upward trend (going viral), others downward (declining)
- **Captured by:** LSTM (nonlinear trends), ARIMA (linear trends), Prophet (piecewise linear)

### Concept: Seasonality
- **What:** Repeating patterns at fixed intervals
- **In our project:** Weekly seasonality in e-commerce (more activity on certain days)
- **Captured by:** Prophet (primary), LSTM can also learn periodic patterns

### Concept: Noise
- **What:** Random, unpredictable fluctuations
- **In our project:** High noise due to synthetic dates + sparse data
- **Handled by:** 7-day smoothing, ensemble averaging (noise averages out across 3 models)

### Concept: Stationarity
- **What:** Statistical properties (mean, variance) don't change over time
- **Required by:** ARIMA
- **How achieved:** d=1 differencing in ARIMA(2,1,2)
- **Q: What if data is non-stationary?**
  - **A:** ARIMA with d=1 automatically differences the data. For LSTM, non-stationarity is handled by MinMax scaling (normalizes to [0,1] range). Prophet decomposes the trend component explicitly.

### Concept: Autocorrelation
- **What:** Correlation of a time series with its own lagged version
- **In our project:** Strong autocorrelation at lag 1-7 (today's mentions predict tomorrow's)
- **Used by:** ARIMA (explicitly models autocorrelation), LSTM (learns it implicitly)

### Concept: Lag Variables
- **What:** Using past values as features (x_t-1, x_t-2, ... x_t-n)
- **In our project:** LSTM's lookback window = 20 lagged values. ARIMA uses 2 lags (p=2).

### Concept: Forecasting Horizon
- **What:** How far into the future we predict
- **In our project:** 15 days
- **Q: Why not 60 days as mentioned in the title (45-60 days)?**
  - **A:** The 45-60 day "early detection" happens through the trend scoring system, not the individual forecast horizon. Trend scores identify products with high growth momentum NOW. By the time they reach peak, it's 45-60 days later. The 15-day forecasting horizon validates the short-term prediction accuracy of the ensemble model.

---

## 13. SOFTWARE ENGINEERING & ARCHITECTURE

### Concept: Pipeline Architecture
- **What:** Sequential processing steps where output of Step N is input of Step N+1
- **In our project:**
  ```
  CSV files → data_loader → augmenter → scorer → forecaster → validator → visualizer
  ```

### Concept: Separation of Concerns
- **What:** Each module handles exactly one responsibility
- **In our project:**
  | Module | Responsibility |
  |--------|---------------|
  | config.py | All constants & paths |
  | data_loader.py | Load, clean, aggregate data |
  | data_augmentation.py | Smoothing + synthetic generation |
  | trend_scorer.py | 4-factor scoring |
  | forecasting_model.py | LSTM + ARIMA + Prophet ensemble |
  | validator.py | Compute metrics |
  | aggregate_validator.py | Market-level validation |
  | visualizer.py | Generate plots |
  | main.py | Orchestrate everything |

### Concept: Configuration Management
- **What:** Centralizing all tunable parameters
- **In our project:** config.py holds ALL hyperparameters
- **Why:** Reproducibility (anyone can see exact settings), easy tuning (change one file → affects whole pipeline)

### Concept: Error Handling
- **In our project:** try/except around ARIMA and Prophet (they can fail on edge cases). If one model fails → returns zeros → other two models compensate.
- **Q: What happens if ARIMA fails?**
  - **A:** Returns array of zeros. The ensemble becomes (0.55×LSTM + 0.30×0 + 0.15×Prophet) / (0.55+0.15) — effectively LSTM + Prophet weighted average. Graceful degradation.

### Concept: PyTorch vs TensorFlow
- Both are deep learning frameworks with similar capabilities
- **PyTorch chosen because:** TensorFlow has Windows CUDA/GPU installation conflicts. PyTorch integrates cleanly on Windows.
- **Q: Would TensorFlow give different results?**
  - **A:** No. Both implement the same mathematical operations. LSTM is LSTM regardless of framework. The neural network architecture and math are identical.

---

## 14. DATA SOURCES & DATASET KNOWLEDGE

### Amazon Sales Dataset (Kaggle)
- **Source:** kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
- **~1,400 products, ~19K reviews**
- **Raw Columns:** product_id, product_name, category, discounted_price, actual_price, discount_percentage, rating (1-5), rating_count, about_product, user_id, user_name, review_id, review_title, review_content, img_link, product_link
- **What we use:** product_name → product, rating → sentiment (÷5), count → mentions
- **What we DON'T use:** prices, images, links, review text (potential future work)

### Flipkart Products Dataset (Kaggle)
- **Source:** kaggle.com/datasets/PromptCloudHQ/flipkart-products
- **~20K products**
- **Raw Columns:** uniq_id, crawl_timestamp, product_url, product_name, product_category_tree, pid, retail_price, discounted_price, image, is_FK_Advantage_product, description, product_rating, overall_rating, brand, product_specifications
- **What we use:** product_name → product, product_rating → sentiment (÷5), count → mentions
- **What we DON'T use:** prices, images, descriptions, specifications

### Processed Data (After Our Pipeline)
| Column | Type | Range | Source |
|--------|------|-------|--------|
| date | datetime | 180-day window | Synthetic (random assignment) |
| product | string | ~14K+ products | From product_name |
| mentions | float | 0.1-50+ | COUNT of reviews, smoothed |
| sentiment | float | 0-1 | rating/5.0 |
| source | string | amazon/flipkart/augmented | Origin identifier |

- **Q: Can you tell me every column in your data and where it comes from?**
  - **A:** (See table above. Know this cold.)

---

## 15. VISUALIZATION & INTERPRETATION

### 13 Plots Generated
1. **Data Source Overview** — Pie chart (records per source) + bar chart (products per source)
2. **Daily Trends** — Total mentions over time with 7-day MA
3. **Sentiment Distribution** — Histogram + boxplot + time series of sentiment per source
4. **Top Products** — Bar charts of top 15 by volume and by data coverage
5. **Data Quality Metrics** — Summary statistics
6. **Mentions Histogram** — Distribution of daily mention counts
7. **Weekly Patterns** — Day-of-week patterns
8. **Correlation Heatmap** — Relationship between mentions, sentiment, trend scores
9. **Trend Leaderboard** — Top 15 products by trend score with color-coded bars
10. **Score Components** — Stacked bar showing 4-factor breakdown per product
11. **Forecast vs Actual** — Primary result: actual test data vs ensemble prediction with 95% CI
12. **Validation Metrics** — MAPE/MAE/RMSE summary with PASS/FAIL indicators
13. **Ensemble Components** — Individual LSTM/ARIMA/Prophet predictions vs ensemble

### Concept: Matplotlib/Seaborn
- **Matplotlib:** Python's core plotting library. Low-level control.
- **Seaborn:** Built on matplotlib. Statistical visualization with better defaults.
- **In our project:** Both used. Seaborn for style (`whitegrid`), matplotlib for fine control.

---

## 16. FAILURE CASES & LIMITATIONS

### Failure Case 1: Sparse Per-Product Data
- **Problem:** Most products have <30 data points → too few for LSTM (needs lookback 20 + horizon 15 + sequences)
- **Solution:** Aggregate market validation (all products combined → dense time series)
- **Impact:** Individual product accuracy varies. Aggregate accuracy is reliable (~87%).

### Failure Case 2: Synthetic Dates
- **Problem:** Kaggle data lacks real review timestamps
- **Solution:** Random date assignment + smoothing + aggregation (averaging removes randomness)
- **Impact:** Individual product timelines are noisy. Aggregate trend emerges correctly.

### Failure Case 3: Cold Start
- **Problem:** New products with zero history → can't forecast
- **Solution:** NONE (acknowledged limitation). Future work: content-based features.

### Failure Case 4: External Shocks
- **Problem:** Unpredictable events (celebrity endorsement, scandal, supply disruption)
- **Solution:** 95% confidence interval captures some uncertainty. Future: external signal integration.

### Failure Case 5: Flat LSTM Predictions
- **Problem:** LSTM outputs constant value when training data is pure noise
- **Solution:** Data augmentation (synthetic products with real trends) + smoothing

### Known Limitations
1. Synthetic dates — biggest limitation
2. No real-time data — validated on static datasets
3. Fixed ensemble weights — ideally should be learned
4. Only review signals — no pricing, social media, search trends
5. Scalability — works on ~14K products, not tested on millions
6. No NLP — star ratings only, no review text analysis
7. Windows-only testing — not tested on Linux/Mac

---

## 17. ACADEMIC CONTEXT & LITERATURE

### Key References
| # | Paper | Year | Relevance |
|---|-------|------|-----------|
| 1 | Hochreiter & Schmidhuber, "Long Short-Term Memory" | 1997 | LSTM architecture |
| 2 | Box & Jenkins, "Time Series Analysis" | 1970 | ARIMA foundations |
| 3 | Taylor & Letham, "Forecasting at Scale" | 2018 | Facebook Prophet |
| 4 | Makridakis et al., "M4 Competition" | 2020 | Ensemble superiority |
| 5 | Hu & Liu, "Mining and Summarizing Customer Reviews" | 2004 | Rating-based sentiment |
| 6 | Kingma & Ba, "Adam: A Method for Stochastic Optimization" | 2015 | Adam optimizer |
| 7 | Shorten & Khoshgoftaar, "Data Augmentation Survey" | 2019 | Augmentation justification |
| 8 | Bandara et al., "Forecasting Across Time Series Databases" | 2019 | RNN for demand forecasting |
| 9 | Lim et al., "Temporal Fusion Transformers" | 2021 | State-of-the-art comparison |
| 10 | Crane & Sornette, "Robust Dynamic Classes" | 2008 | Viral content lifecycle |
| 11 | Pang & Lee, "Opinion Mining and Sentiment Analysis" | 2008 | Sentiment literature |
| 12 | Hyndman & Athanasopoulos, "Forecasting: Principles & Practice" | 2018 | Time series textbook |

### State of the Art Comparison
| Model | MAPE | Data Type | Comparison |
|-------|------|-----------|------------|
| N-BEATS (Oreshkin, 2020) | ~11% | Dense financial | Better accuracy, but needs 1000s of points per series |
| TFT (Lim, 2021) | ~9% | Multi-horizon | SOTA, but extremely complex architecture |
| DeepAR (Amazon, 2020) | ~15% | Dense retail | Amazon's in-house tool, not publicly replicated |
| **Our Ensemble** | **~12.8%** | **Sparse e-commerce** | **Comparable results on much sparser data** |

---

## 18. BUSINESS VALUE & REAL-WORLD APPLICATION

### The Business Problem
- E-commerce sellers need 45-60 days lead time to source products from manufacturers
- If they can detect a rising trend EARLY, they can stock up before competitors
- Current approach: manual market research (slow, subjective)
- Our approach: automated trend detection from review signals

### How It Would Work in Practice
1. Daily: Scrape new reviews from Amazon/Flipkart
2. Daily: Run pipeline (< 5 minutes)
3. Alert: Products with trend_score > 60 are flagged
4. Seller: Sees "Product X is trending with 87% confidence" → starts sourcing
5. 45-60 days later: Product peaks → seller already has stock → profit

### The Signal Hypothesis
- **Hypothesis:** Review volume and sentiment are LEADING indicators of sales
- **Why?** Early adopters review products before mass market buys
- **Evidence:** Our model predicts peaks with ±0-6 day accuracy
- **Novelty:** Most papers analyze review TEXT. We analyze review TIMING.

---

## 19. MATHEMATICAL FOUNDATIONS

### Linear Algebra
- **Matrix multiplication:** Core of neural networks. Input × Weights + Bias
- **In LSTM:** h_t = σ(W_h · h_{t-1} + W_x · x_t + b)

### Calculus
- **Derivatives:** Gradients for backpropagation
- **Chain rule:** How error propagates through layers: ∂L/∂w = ∂L/∂y × ∂y/∂h × ∂h/∂w
- **Exponential decay:** e^(-kt) for product lifecycle modeling

### Statistics
- **Mean, Variance, Standard Deviation:** Used in normalization, metrics, CI
- **Correlation (Pearson r):** Growth velocity vs peak timing (r=0.65)
- **Confidence Interval:** forecast ± 1.96 × σ (95% CI from z-distribution)

### Probability
- **Sigmoid function:** σ(x) = 1/(1+e^(-x)) — used in LSTM gates
- **Loss as expected error:** MSE = E[(y - ŷ)²]

### Optimization
- **Gradient descent:** w ← w - α × ∇L(w)
- **Adam:** Combines momentum (β₁=0.9) and RMSprop (β₂=0.999)

---

## 20. QUICK-FIRE 50 QUESTIONS

| # | Question | Answer |
|---|----------|--------|
| 1 | What does your project do? | Predicts e-commerce product trends 45-60 days ahead using review signals |
| 2 | What data? | Amazon (~1,400 products) + Flipkart (~20K products) from Kaggle |
| 3 | Total records? | ~32K after augmentation (21K real + 11K synthetic) |
| 4 | What are the columns? | date, product, mentions (count), sentiment (0-1), source |
| 5 | Where does sentiment come from? | sentiment = star_rating / 5.0 |
| 6 | Why not NLP? | Star ratings >95% reliable; NLP on short reviews ~70% |
| 7 | Why synthetic dates? | Kaggle datasets have no review timestamps |
| 8 | What model? | Hybrid ensemble: LSTM(55%) + ARIMA(30%) + Prophet(15%) |
| 9 | Why ensemble? | M4 Competition: ensembles beat single models by 10-15% |
| 10 | What's LSTM? | Long Short-Term Memory — RNN with gates for long-term dependencies |
| 11 | LSTM architecture? | 3-layer: 256→128→64 neurons, dropout 15% |
| 12 | Why 3 layers? | Progressive compression: raw patterns → abstract features → output |
| 13 | What are LSTM gates? | Forget (what to discard), Input (what to store), Output (what to emit) |
| 14 | Why not RNN? | Vanishing gradient — can't learn patterns >10 steps back |
| 15 | What's ARIMA? | AutoRegressive Integrated Moving Average |
| 16 | ARIMA order? | (2,1,2): p=2 lags, d=1 differencing, q=2 MA terms |
| 17 | What's p=2? | Uses last 2 time values to predict current value |
| 18 | What's d=1? | One round of differencing to achieve stationarity |
| 19 | What's q=2? | Uses last 2 forecast errors to self-correct |
| 20 | What's stationarity? | Constant mean and variance over time |
| 21 | What's Prophet? | Facebook's tool for business time series with seasonality |
| 22 | Why multiplicative? | E-commerce effects SCALE with trend level |
| 23 | Why PyTorch? | TensorFlow has Windows CUDA issues; identical math |
| 24 | What optimizer? | Adam (lr=0.0008) |
| 25 | What loss? | MSE (Mean Squared Error) |
| 26 | What's dropout? | Randomly disable 15% of neurons during training to prevent overfitting |
| 27 | What's early stopping? | Stop training if no improvement for 10 epochs |
| 28 | What's MinMaxScaler? | Normalizes values to [0,1] range for LSTM |
| 29 | What's lookback? | 20 days of history used as LSTM input |
| 30 | What's forecast horizon? | 15 days predicted output |
| 31 | How is trend score calculated? | growth(40%) + sentiment(20%) + saturation(20%) + profit(20%) |
| 32 | Why growth 40%? | Strongest predictor — r=0.65 correlation with peaks |
| 33 | What's saturation index? | 1 - (current/max) — penalizes products past their peak |
| 34 | What's profit potential? | Second derivative of growth (acceleration) |
| 35 | What's MAPE? | Mean Absolute Percentage Error — average % prediction error |
| 36 | Your MAPE? | ~12.80% |
| 37 | Your accuracy? | ~87.20% (= 100 - MAPE) |
| 38 | Target accuracy? | >70% (MAPE <30%) |
| 39 | Peak error? | ±0-6 days (target: ±7) |
| 40 | What's MAE vs RMSE? | MAE = absolute; RMSE = penalizes big errors more (squared) |
| 41 | Train/test split? | 200 days train, 15 days test (chronological, no random) |
| 42 | What augmentation? | 7-day smoothing + 30 synthetic products (360 days each) |
| 43 | Why augmentation? | LSTM needs more training sequences; real data too sparse |
| 44 | Failure cases? | Sparse data, synthetic dates, cold start, external shocks, flat LSTM |
| 45 | Biggest limitation? | Synthetic dates — Kaggle data has no real timestamps |
| 46 | What's novel? | Review TIMING as early indicator + ensemble on sparse data |
| 47 | Business value? | Detect trends 45-60 days before peak → source inventory early |
| 48 | How many parameters? | ~500K+ (LSTM trainable weights) |
| 49 | Confidence interval? | 95% CI = forecast ± 1.96 × std |
| 50 | How to improve? | Add NLP, real-time scraping, learned weights, Transformer architecture |

---

## INSTRUCTOR-SPECIFIC CONCERNS (From Mid-Sem Feedback)

### "They could not explain the data columns"
→ **Memorize Section 14.** Know every column, its type, range, and exactly how it's computed.

### "They could not explain the models and specific implementation"
→ **Memorize Sections 2, 3, 4.** Know LSTM gates, ARIMA (p,d,q), Prophet seasonality.

### "Why is that you're taking only 20% weightage here?"
→ Each factor measures a DIFFERENT aspect. Sentiment (20%) validates trend quality. Remaining 80%: Growth(40%) captures speed, Saturation(20%) captures remaining potential, Profit(20%) captures acceleration. Total = 100%.

### "What self-weight you're adding? Why?"
→ Every weight is justified. Growth 40% because r=0.65 with peaks. Testing showed 40/20/20/20 beats equal 25/25/25/25 by 8%.

### "I'm purely looking at your knowledge of data science"
→ **Know:** overfitting vs underfitting, bias-variance tradeoff, train-test split rationale, stationarity, scaling, dropout mechanism, early stopping logic, ensemble theory, MAPE interpretation, gradient descent, backpropagation.

### "How did you fine-tune? MSE different than expected."
→ Early stopping (patience=10), learning rate tuned from 0.001 → 0.0008, dropout tested 10/15/20%, lookback tested 10/20/30, ARIMA order selected by AIC. Data augmentation added 10,800 synthetic records.

### "What specific failure case analysis did you do?"
→ **Memorize Section 16.** Five failure cases with root causes and solutions. Plus sparsity analysis and cold start acknowledgment.

---

*Last updated: Feb 2026. Keep this document with your project files.*
