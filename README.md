# Market_Pulse_Forecast

## 📈 Overview

Build a stock prediction app using an LSTM (Long Short-Term Memory) network to forecast price movements for volatile stocks. Integrate historical price data, news sentiment, and technical indicators to enhance prediction accuracy. The model leverages multiple market factors to make informed predictions, considering both short-term and long-term trends. Emphasis is placed on understanding the key drivers, such as sentiment scores and price trends, that influence stock movements.

The goal is to provide better insights for investors and traders by accurately finding features that cause stock price fluctuations.

---

## 💹 Why Volatile Stock?

Choose **MSFT** for this prediction model due to its price volatility. Volatile stocks present more movement, which can be influenced by factors such as news sentiment, market conditions, and global events. This allows you to evaluate how well the model generalizes when learning from dynamic and complex data.

---

## 1. 📦 Data Collection

Ensure the accuracy of stock price prediction by leveraging high-quality, diverse data. Use a combination of:

- **Historical Stock Data**: Open, close, high, low, and volume — foundational for prediction.
- **News Data**: Headlines and articles are scraped to capture sentiment, which can drive price movement.
- **Technical Indicators**: Moving averages, volatility, and RSI for quantitative signals.

### 🔍 News Scraping & APIs

Use **financialmodelingprep pro** (e.g., NewsAPI, Alpha Vantage, or other open-source API) to collect news related to the target stock.

- Free access to real-time and historical news
- Wide range of reputable sources
- Filter by keywords, tickers, and stock-related tags

### 🧹 Preprocessing & Structuring

Preprocess both stock and news data:

- **Stock**: Handle missing values, normalize, calculate moving averages, volatility, etc.
- **News**: Clean headlines, apply sentiment scoring (e.g., VADER, TextBlob)

### 🔗 Merging Data

Merge sentiment and stock data by aligning timestamps (daily granularity). Final dataset includes:

- Historical stock prices (OHLCV)
- News sentiment scores
- Technical indicators
- Calendar features (day of week, month)

---

## 2. 🧠 Model Training & Evaluation

### 📌 Model Selection & Training

Use **LSTM** due to its strength in modeling time dependencies. The model is trained on:

- Previous stock prices
- Sentiment scores from news articles
- Technical features (e.g., moving averages)

### ⚙️ Hyperparameter Tuning

Tune parameters like:

- Number of layers and LSTM units
- Optimizer and learning rate
- Batch size and number of epochs

Use **Grid Search** or **Random Search** for systematic experimentation.

### 📊 Model Validation & Testing

Evaluate performance using:

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Baseline Comparison**: Compare with a naive model using only historical price data.

---

## 3. 🧠 Explainability Methods

Interpret the model using:

### 🔄 Counterfactual Explanations

Show "what-if" scenarios by tweaking inputs (e.g., changing sentiment) and observing how predictions shift.

### 📌 Feature Importance

Measure how each feature (sentiment, technicals, prices) impacts predictions via SHAP or permutation importance.

### 🧪 LIME

Explain individual predictions using **LIME** to show which features influenced decisions at specific time steps.

### 🧠 Attention & Saliency Maps

Use attention or saliency maps to visualize which time steps or features mattered most during prediction.

### 🌈 SHAP

Apply **SHAP** (Shapley values) to quantify each feature's contribution to model output.

---

## 4. 🧪 Time Series Data Splitting Strategies

Implement multiple splitting strategies for robust model evaluation:

### 1. Chronological Split
- Train on older data, test on newer
- Mimics real-world forecast flow
- Simple but limited to one test set

### 2. Expanding Window
- Gradually increase training size
- Reflects a growing knowledge base
- Good for model learning curves

### 3. Time Series Cross-Validation (k-Fold)
- Multiple train/test splits in order
- Avoids data leakage
- Reliable and thorough but time-consuming

### 4. Walk-Forward (Rolling Window)
- Fixed training window slides forward
- Reacts to market changes
- Requires retraining each step


### What is good and when

- chronological	Fastest and good for baseline
- expanding	Want to simulate growing market knowledge
- rolling	Want recency-sensitive models (walk-forward test)
- kfold	Want model robustness, good stats


---

## 5. 🧱 Modular Project Design

Please make sure to run the **scripts** folder to prepare the data first.

Designed for experimentation and reusability:

<pre> market_pulse_forecast/ 
  ├── data/ 
  │     ├── loader.py                # Load and preprocess stock + sentiment data 
  │     └── features.py              # Create & select technical and calendar features  
  ├── splits/ 
  │   └── splitter.py                # All time series splitting strategies 
  ├── models/ 
  │   └── lstm_model.py              # LSTM model definition with attention option 
  ├── training/ 
  │   └── trainer.py                 # Model training, evaluation, logging  
  ├── explainability/ 
  │   ├── shap_explainer.py          # Use SHAP to compute feature contribution scores
  │   ├── lime_explainer.py          # Use LIME to explain individual model predictions
  │   ├── saliency.py                # Visualize which time steps/inputs affect predictions
  │   └── counterfactuals.py         # Generate what-if scenarios to explore prediction shifts
  ├── tuning/ 
  │   └── search.py                  # Grid/Random search for hyperparameters
  ├── config/ 
  │   └── settings.yaml              # Feature selection, model config, etc. 
  │ 
  ├── data_prep.py                   # Based on stock ticker (settings.yaml), prepares your data
  │  
  └── run.py                         # Entry point for training & experimentation </pre>



## Execution

### Prepare data

python run.py --mode data-prep

### Train and evaluation

<details>
  <summary>Train and Evaluation default </summary>
  
  - python run.py --mode train
  - python run.py --mode evaluate
</details>

(by default, it will choose chronological, to change please use below format)

<details>
  <summary>Train and Evaluation with other split methods </summary>

  - python run.py --mode train --split_method expanding
  - python run.py --mode evaluate --split_method expanding
  
  - python run.py --mode train --split_method rolling
  - python run.py --mode evaluate --split_method rolling
  
  - python run.py --mode train --split_method kfold
  - python run.py --mode evaluate --split_method kfold
</details>




### Tuning

python run.py --mode tune


### Explanantion

<details>
  <summary>Pass explanantion methods </summary>
  
  - python run.py --mode explain --method shap
  - python run.py --mode explain --method lime
  - python run.py --mode explain --method saliency
  - python run.py --mode explain --method counterfactual

</details>


## 🔐 Verifiable ML Pipeline with Blockchain

This blockchain-based approach ensures end-to-end transparency and reproducibility of the ML pipeline by immutably logging hashes of critical artifacts. Instead of storing raw data, only cryptographic hashes of configurations, model states, and results are sent to the Sepolia testnet. Each hash acts as a verifiable fingerprint, guaranteeing that no step was altered post-recording. This method adds a layer of integrity and auditability without exposing sensitive or large datasets. It’s a lightweight yet powerful way to bring accountability into machine learning workflows.



| #  | Checkpoint Name               | Description                                                            | Source for Hashing             |
|----|-------------------------------|------------------------------------------------------------------------|--------------------------------|
| 0  | config_settings               | Model + data settings from YAML                                        | settings.yaml                  |
| 1  | raw_data_hash                 | Raw financial + sentiment data (in-memory snapshot)                    | df.to_json(orient="split")     |
| 2  | feature_flags_hash            | Feature config from YAML                                               | settings.yaml["feature_flags"] |
| 3  | split_strategy_hash           | Train/test split metadata (in-memory)                                  | metadata dict from splitter    |
| 4  | train_dataset_hash            | Processed training set (in-memory)                                     | train_df.to_json()             |
| 5  | val_dataset_hash              | Processed validation/test set                                          | test_df.to_json()              |
| 6  | tuned_hyperparams_hash        | Final tuning params (as dict or json)                                  | tuning_result dict             |
| 7  | model_weights_hash            | Raw model weights (in-memory bytes)                                    | model.get_weights() or .state_dict() |
| 8  | training_metrics_hash         | Training loss/metrics per epoch                                        | history/history_callback       |
| 9  | evaluation_metrics_hash       | Final evaluation scores                                                | metrics dict                   |
|10  | attention_weights_hash        | If attention mechanism is used                                         | attention_output ndarray       |
|11  | shap_explanation_hash         | SHAP explanation values (e.g. summary values)                          | shap_values.to_json()          |
|12  | attention_plot_hash           | Attention heatmap as image in-memory                                   | BytesIO image buffer           |
|13  | final_predictions_hash        | Model predictions vs true values                                       | predictions.to_json()          |






### Protocol handling approach

Hash each of these (from memory or temp object), send it to Sepolia, and log the returned transaction hash under the corresponding key.

```python
lstm_explanation_protocol = {
    "config_settings": tx_hash_hex_0,
    "raw_data_hash": tx_hash_hex_1,
    "feature_flags_hash": tx_hash_hex_2,
    ...
    "final_predictions_hash": tx_hash_hex_13
}
```


















---

© 2025 SB. All rights reserved.
