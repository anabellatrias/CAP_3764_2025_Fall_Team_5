Gold vs. S&P 500 Outperformance Prediction

Team 5: Anabella Trias and Anthony Sierra

Predicting 90-day gold outperformance using macroeconomic time-series data (2006–2024)

Overview

This project builds a binary classification model that predicts whether gold will outperform the S&P 500 over a 90-day horizon, using 18 years of macroeconomic and financial indicators.

The goal is to explore feature engineering, time-series modeling, and evaluation strategies for tactical asset-allocation signals.

---

Project Structure

/data               
/notebooks
    01_data_processing.ipynb
    02_eda.ipynb
    03_modeling.ipynb
data_pipeline.py    
README.md
requirements.txt

Data Sources

Combined 11 datasets from Investing.com and FRED (2006–2024), including:
- Gold spot price
- S&P 500 index
- CPI (inflation)
- Crude oil
- Federal funds rate
- VIX (volatility index)
- Treasury yields (2Y, 10Y)
- USD index
- Labor data
- Recession indicators

Final processed dataset: 4,000+ daily observations.

⸻

Problem Definition

Target:
Gold outperforms the S&P 500 over the next 90 days (1 if true, 0 otherwise).

Type:
Binary classification on time-dependent data.

---

Feature Engineering

Created 40+ features using pandas and NumPy, including:

Market & Price Features
- Daily & rolling returns
- 30/60/90-day momentum
- Rolling means (7/14/30 days)
- Rolling volatility
- Drawdown metrics

Macroeconomic Indicators
- Inflation changes
- Yield curve slope
- USD strength
- Oil price fluctuations
- Volatility spikes (VIX)

Target Construction
- 90-day forward relative return:
gold_future_return - sp500_future_return

---

Exploratory Analysis

Key patterns identified:
- Gold’s strongest relative performance spikes align with crisis periods (2008, 2020).
- Increasing VIX, falling yields, and USD volatility strongly correlate with gold outperformance.
- Distribution shapes vary significantly pre- and post-2013.

---

Modeling Approach

Models Evaluated
- Logistic Regression (balanced)
- Random Forest
- XGBoost

Time-Based Split

To prevent leakage:
- Train: earlier years
- Test: later years

Performance (Best Model)

Logistic Regression (balanced):
- ROC AUC: 0.753
- Recall: 82.6%
- Precision: 59.2%
- Prioritizes recall to maximize signal detection in tactical allocation strategies.

---

In-Progress Work
- Benchmarking with PyCaret to compare automated ML workflows against manual baselines
- Creation of Streamlit/FastAPI dashboard 

---

Tech Stack
	•	Python (pandas, NumPy, scikit-learn)
	•	Matplotlib / Seaborn
	•	Jupyter Notebooks
	•	Conda environment
