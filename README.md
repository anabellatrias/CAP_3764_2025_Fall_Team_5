# Gold vs. S&P 500 Outperformance Prediction

**Team 5:** Anabella Trias and Anthony Sierra  
**Course:** CAP 3764 - Advanced Data Science, Fall 2025

Predicting 90-day gold outperformance using macroeconomic time-series data (2006–2024)

---

## 📋 Overview

This project builds a binary classification model that predicts whether **gold will outperform the S&P 500 over a 90-day horizon**, using 18 years of macroeconomic and financial indicators.

**Business Question:** Should investors allocate capital to gold or stocks over the next 90 days?

**Target Variable:** Binary classification (1 = gold outperforms, 0 = stocks outperform)

---

## 🚀 How to Run This Project

### Prerequisites
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd CAP_3764_2025_Fall_Team_5-5

# 2. Create conda environment
conda env create -f gold_prediction_environment.yml
conda activate gold_predictor

# 3. Run all Notebooks

# 4. Run api.py file
python api.py

# 5. Run streamlit app
streamlit run app.py

### Run the Analysis Pipeline

**Step 1: Data Processing**
```bash
jupyter notebook notebooks/01_data_processing.ipynb
```
- Loads 11 raw datasets from `data/raw/`
- Merges daily market data with monthly macro indicators
- Engineers 40+ features (returns, volatility, moving averages)
- Creates 90-day forward target variable
- **Output:** `data/processed/gold_features.csv` (4,548 observations)

**Step 2: Exploratory Data Analysis**
```bash
jupyter notebook notebooks/02_eda.ipynb
```
- Visualizes gold vs. stocks performance over time
- Analyzes correlation between features and target
- Identifies crisis periods (2008, 2020) where gold dominated
- **Output:** Visualizations saved to `reports/figures/`

**Step 3: Model Training & Evaluation**
```bash
jupyter notebook notebooks/03_modeling.ipynb
```
- Implements time-based train/test split (2006-2017 train, 2017-2020 test)
- Trains Extra Trees Classifier using PyCaret
- **Cross-validation:** 10-fold stratified CV on training set
- Evaluates on held-out test set
- **Output:** 
  - Trained model saved to `models/`
  - Performance metrics (CV AUC: 0.964, Test AUC: 0.775)

### Run the Deployment

**Start the API and Web App:**
```bash
# Terminal 1 - Start FastAPI backend
cd deployment
python api.py

# Terminal 2 - Start Streamlit frontend
cd deployment
streamlit run app.py
```

**Access the application:**
- Web Interface: http://localhost:8501
- API Documentation: http://localhost:8000/docs

---

## 📁 Project Structure

```
├── data/
│   ├── raw/                    # Original datasets from Investing.com & FRED
│   └── processed/              # Cleaned and merged data with engineered features
├── notebooks/
│   ├── 01_data_processing.ipynb   # Data loading, cleaning, feature engineering
│   ├── 02_eda.ipynb               # Exploratory analysis and visualizations
│   └── 03_modeling.ipynb          # Model training, CV, evaluation
├── models/
│   ├── gold_predictor_pipeline.pkl  # Trained Extra Trees model
│   ├── feature_names.pkl            # List of input features
│   └── model_metadata.json          # Model performance metrics
├── deployment/
│   ├── api.py                   # FastAPI backend for predictions
│   ├── app.py                   # Streamlit web interface
│   └── sample_batch.csv         # Sample data for batch predictions
├── src/
│   └── data_pipeline.py         # Reusable data processing functions
└── README.md                    # This file
```

---

## 📊 Data Sources

Combined **12 datasets** from Investing.com and FRED (2006–2024):

**Daily Market Data (Investing.com):**
- Gold futures prices
- S&P 500 index
- Silver futures
- USD Index (DXY)
- VIX (volatility index)
- NASDAQ index
- Crude oil prices

**Monthly Macroeconomic Indicators (FRED):**
- CPI (inflation)
- Federal Funds Rate
- Unemployment Rate
- M2 Money Supply
- US Treasury Yields (1Y, 10Y)

**Final Dataset:** 4,548 daily observations spanning 2006-2024

---

## 🔧 Feature Engineering

Created **40+ features** organized into five categories:

### 1. Price & Returns
- Daily prices for gold, S&P 500, silver, oil
- 5-day, 20-day, 60-day returns
- Rolling momentum indicators

### 2. Moving Averages
- 20-day moving averages for all assets
- Price-to-MA ratios

### 3. Volatility Metrics
- 20-day and 60-day rolling volatility (annualized)
- Volatility regime indicators

### 4. Macroeconomic Indicators
- CPI year-over-year change
- Real interest rate (Fed Funds - Inflation)
- Yield curve slope (10Y - 2Y)
- M2 money supply growth

### 5. Derived Features
- Gold/Silver ratio
- USD strength vs. gold
- VIX regime indicators

**Target Construction:**
```python
# 90-day forward relative return
gold_90d_return = (gold_price_t+90 / gold_price_t) - 1
sp500_90d_return = (sp500_price_t+90 / sp500_price_t) - 1
target = 1 if gold_90d_return > sp500_90d_return else 0
```

---

## 📈 Exploratory Analysis

Key patterns identified in `02_eda.ipynb`:

1. **Crisis Performance**
   - Gold strongly outperformed during 2008 financial crisis and 2020 COVID crash
   - Stocks dominated during expansion periods (2013-2019)

2. **Feature Correlations**
   - High VIX → Gold outperformance
   - Rising interest rates → Stock outperformance
   - USD weakness → Gold strength

3. **Target Distribution**
   - 42.8% gold wins, 57.2% stocks win

---

## 🤖 Modeling Approach

### Models Evaluated
Tested multiple algorithms using PyCaret's automated ML workflow:

1. **Logistic Regression** (baseline)
2. **Random Forest**
3. **Extra Trees Classifier** ⭐ (best model)
4. **XGBoost**
5. **LightGBM**

### Cross-Validation Strategy

**Implemented in:** `03_modeling.ipynb` using PyCaret's `setup()` and `compare_models()`

- **Method:** 10-fold stratified cross-validation
- **Metric:** ROC AUC (Area Under ROC Curve)
- **Location:** CV results displayed after model tuning cell

**Note on Validation:**
We used stratified K-fold CV as implemented by PyCaret. For future improvements, we acknowledge that time-series cross-validation (walk-forward) would better respect temporal dependencies.

### Train/Test Split

**Critical for time-series data:** Used temporal split to prevent lookahead bias

- **Training:** 2006-2017 (2,766 observations)
- **Testing:** 2017-2020 (1,782 observations)
- **Validation:** No data from the future is used to predict the past

### Best Model Performance

**Extra Trees Classifier (Tuned):**

| Metric | Cross-Validation (10-fold) |
|--------|---------------------------|
| ROC AUC | 0.9897 ± 0.007 |
| Accuracy | 95.1% |
| Precision | 94.0% |
| Recall | 94.7% |
| F1 Score | 94.3% |

**Note:** Test set performance showed perfect classification (AUC = 1.0), indicating potential data leakage. Cross-validation metrics are more reliable for evaluating model performance.

**Why Extra Trees?**
- Handles non-linear relationships between features
- Robust to overfitting through randomization
- Fast training and prediction
- Built-in feature importance

---

## 🚀 Deployment

### FastAPI Backend (`deployment/api.py`)

RESTful API for model predictions:

**Endpoints:**
- `GET /` - API information
- `POST /predict` - Single prediction
- `GET /docs` - Interactive API documentation

**Example API Usage:**
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "gold_price": 1850.0,
        "sp500_price": 4200.0,
        "silver_price": 23.5,
        "usd_index_value": 103.0,
        "treasury_yield": 4.5,
        "nasdaq_value": 13500.0,
        "vix_value": 15.0,
        "oil_price": 78.0
    }
)
print(response.json())
# Output: {"prediction": "Stocks", "probability": "44.0%", "winner": "📈 Stocks will outperform"}
```

### Streamlit Frontend (`deployment/app.py`)

Interactive web interface with two modes:

1. **Single Prediction**
   - Enter current market values in form
   - Get instant prediction with probability
   - See investment recommendation

2. **Batch Predictions**
   - Upload CSV with multiple scenarios
   - Process all predictions at once
   - Download results as CSV

**Features:**
- Real-time predictions
- Visual probability display
- Sample data download
- Results export

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Data Processing:** pandas, NumPy
- **Machine Learning:** scikit-learn, PyCaret, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **API:** FastAPI, Uvicorn
- **Frontend:** Streamlit
- **Development:** Jupyter Notebooks, Conda

---

## 📊 Key Results

### Model Performance Summary

Our Extra Trees model achieved strong performance on both validation and test sets:

- **Cross-Validation AUC: 0.964** (10-fold CV on training data)
- **Test Set AUC: 0.775** (held-out 2017-2020 data)
- **Recall: 79.8%** - Successfully identifies most gold outperformance periods
- **Precision: 76.5%** - High confidence when predicting gold wins

### Feature Importance

Top predictive features (from Extra Trees model):
1. Gold/Silver ratio
2. Real interest rate (Fed Funds - CPI)
3. VIX 60-day volatility
4. Gold 20-day return
5. Yield curve slope (10Y - 2Y)

### Business Value

The model provides actionable signals for tactical asset allocation:
- **High confidence predictions** (probability > 60% or < 40%) can guide portfolio rebalancing
- **Crisis detection** - Model successfully identifies gold-favoring regimes
- **Risk management** - Helps investors hedge equity exposure during volatile periods

---

## 🔄 Workflow Summary

```
Raw Data (11 sources)
    ↓
[01_data_processing.ipynb]
    ↓
Processed Dataset (4,548 observations, 40+ features)
    ↓
[02_eda.ipynb] → Insights & Visualizations
    ↓
[03_modeling.ipynb] → Cross-Validation → Model Selection → Test Evaluation
    ↓
Saved Model (models/)
    ↓
[deployment/api.py] ← FastAPI Backend
    ↓
[deployment/app.py] ← Streamlit Frontend
    ↓
Production Predictions
```

---

## 📝 Limitations & Future Work

### Current Limitations

1. **Validation Method**
   - Used stratified K-fold CV which doesn't respect temporal order
   - Future: Implement walk-forward validation for time-series

2. **Feature Engineering**
   - Limited to technical and macro indicators
   - Future: Add sentiment analysis, geopolitical events, commodity correlations

3. **Model Scope**
   - Trained on 2006-2020 data
   - Future: Regular retraining to adapt to changing market regimes

### Planned Improvements

- [ ] Time-series cross-validation (walk-forward)
- [ ] Additional features (sentiment, seasonality)
- [ ] Ensemble of multiple models
- [ ] Real-time data integration
- [ ] Trading strategy backtesting with transaction costs

---

## 👥 Team Contributions

**Anabella Trias:**
- Data processing pipeline development
- Data Cleaning and EDA
- Feature engineering
- Deployment (FastAPI + Streamlit)

**Anthony Sierra:**
- Data collection
- Model comparison and selection
- Model training and evaluation
- Documentation
---

## 📚 References

**Data Sources:**
- Investing.com - Historical market data
- Federal Reserve Economic Data (FRED) - Macroeconomic indicators

**Libraries & Frameworks:**
- PyCaret: https://pycaret.org/
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://streamlit.io/

---
