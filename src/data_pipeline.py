"""
Data Pipeline for Gold Investment Prediction

This module processes 11 datasets (daily market + monthly macro) into a unified 
feature matrix for predicting gold vs S&P 500 outperformance.

Key components:
- Data loading (Investing.com and FRED formats)
- Feature engineering (returns, moving averages, volatility)
- Target creation (90-day forward-looking classification)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_daily_data(filepath):
    """
    Load daily market data from Investing.com or FRED format.
    
    Handles:
    - Comma-separated numbers (e.g., "3,388.50")
    - Percentage strings (e.g., "0.89%")
    - Multiple date column names (Date, DATE)
    
    Args:
        filepath: Full path to CSV file
        
    Returns:
        DataFrame with cleaned numeric columns and datetime index
    """
    df = pd.read_csv(filepath)
    
    # Handle different date column names
    date_col = None
    for col in ['Date', 'DATE', 'date', 'observation_date']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError(f"No date column found in {filepath}")
    
    # Parse dates
    df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.drop(columns=[date_col])
    
    # Clean numeric columns (remove commas, handle percentages)
    for col in df.columns:
        if col == 'date':
            continue
            
        if df[col].dtype == 'object':
            # Remove commas and convert to float
            df[col] = df[col].astype(str).str.replace(',', '')
            
            # Handle percentage signs
            if df[col].str.contains('%', na=False).any():
                df[col] = df[col].str.replace('%', '').astype(float) / 100
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop unnecessary columns (OHLC, Volume, Change %)
    drop_cols = ['Open', 'High', 'Low', 'Vol.', 'Change %']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def load_monthly_data(filepath):
    """
    Load monthly macroeconomic data from FRED format.
    
    Expected format:
        observation_date,VALUE
        1947-01-01,21.48
        
    Args:
        filepath: Full path to CSV file
        
    Returns:
        DataFrame with datetime index and numeric value column
    """
    df = pd.read_csv(filepath)
    
    # FRED uses 'observation_date' column
    if 'observation_date' not in df.columns:
        raise ValueError(f"Expected 'observation_date' column in {filepath}")
    
    df['date'] = pd.to_datetime(df['observation_date'])
    df = df.drop(columns=['observation_date'])
    
    # Convert value columns to numeric
    for col in df.columns:
        if col != 'date':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_values('date').reset_index(drop=True)
    
    return df


def standardize_columns(df, prefix):
    """
    Add prefix to all columns except 'date'.
    
    Args:
        df: DataFrame to standardize
        prefix: Prefix to add (e.g., 'gold', 'sp500')
        
    Returns:
        DataFrame with standardized column names
    """
    rename_dict = {col: f'{prefix}_{col.lower()}' 
                   for col in df.columns if col != 'date'}
    
    return df.rename(columns=rename_dict)


def forward_fill_monthly(daily_df, monthly_df, prefix):
    """
    Merge monthly data into daily dataframe with forward-fill.
    
    Args:
        daily_df: DataFrame with daily frequency
        monthly_df: DataFrame with monthly frequency
        prefix: Prefix for monthly columns
        
    Returns:
        DataFrame with monthly values forward-filled to daily
    """
    # Merge on date with outer join
    merged = daily_df.merge(monthly_df, on='date', how='left')
    
    # Forward-fill monthly values
    for col in monthly_df.columns:
        if col != 'date':
            merged[col] = merged[col].ffill()
    
    return merged

def create_returns(df, periods=[5, 20, 60]):
    """
    Calculate returns over multiple periods.
    
    Args:
        df: DataFrame with price columns
        periods: List of lookback periods
        
    Returns:
        DataFrame with return columns added
    """
    price_cols = [col for col in df.columns if 'price' in col.lower() 
                  and 'ma' not in col.lower()]
    
    for col in price_cols:
        base_name = col.replace('_price', '')
        
        for period in periods:
            # Calculate return: (price_t - price_t-n) / price_t-n
            df[f'{base_name}_return_{period}d'] = (
                df[col].pct_change(periods=period) * 100
            )
    
    return df


def create_moving_averages(df, windows=[20]):
    """
    Calculate moving averages.
    
    Args:
        df: DataFrame with price columns
        windows: List of MA windows (in days)
        
    Returns:
        DataFrame with MA columns added
    """
    price_cols = [col for col in df.columns if 'price' in col.lower() 
                  and 'ma' not in col.lower()]
    
    for col in price_cols:
        base_name = col.replace('_price', '')
        
        for window in windows:
            df[f'{base_name}_ma_{window}'] = (
                df[col].rolling(window=window, min_periods=window).mean()
            )
    
    return df


def create_volatility(df, windows=[20, 60]):
    """
    Calculate rolling volatility (annualized).
    
    Args:
        df: DataFrame with price columns
        windows: List of volatility windows
        
    Returns:
        DataFrame with volatility columns added
    """
    price_cols = [col for col in df.columns if 'price' in col.lower() 
                  and 'ma' not in col.lower() and 'vol' not in col.lower()]
    
    for col in price_cols:
        base_name = col.replace('_price', '')
        
        for window in windows:
            # Calculate daily returns
            returns = df[col].pct_change()
            
            # Rolling standard deviation, annualized
            df[f'{base_name}_vol_{window}d'] = (
                returns.rolling(window=window).std() * np.sqrt(252) * 100
            )
    
    return df


def create_macro_features(df):
    """
    Create derived macroeconomic features.
    
    Args:
        df: DataFrame with macro columns
        
    Returns:
        DataFrame with derived features added
    """
    # Real interest rate (Fed Funds - CPI YoY)
    fedfunds_col = [c for c in df.columns if 'fedfunds' in c.lower()]
    cpi_col = [c for c in df.columns if 'cpi' in c.lower() and 'yoy' not in c.lower()]
    
    if fedfunds_col and cpi_col:
        fedfunds_col = fedfunds_col[0]
        cpi_col = cpi_col[0]
        df['cpi_yoy_change'] = df[cpi_col].pct_change(periods=12) * 100
        df['real_interest_rate'] = df[fedfunds_col] - df['cpi_yoy_change']
    
    # Gold/Silver ratio (using standardized names)
    if 'gold_price' in df.columns and 'silver_price' in df.columns:
        df['gold_silver_ratio'] = df['gold_price'] / df['silver_price']
    
    # Yield curve slope (using treasury column if available)
    treasury_col = [c for c in df.columns if 'treasury' in c.lower() and 'yield' in c.lower()]
    if treasury_col:
        df['yield_curve_slope'] = df[treasury_col[0]]  # Simplified - using 10Y as proxy
    
    return df


def create_target(df, horizon=90, method='outperform_sp500'):
    """
    Create binary target variable.
    
    Args:
        df: DataFrame with price columns
        horizon: Forward-looking window (days)
        method: 'outperform_sp500' or 'price_increase'
        
    Returns:
        DataFrame with 'target' column added
    """
    if method == 'outperform_sp500':
        # Calculate forward returns
        df['gold_return_forward'] = df['gold_price'].pct_change(periods=horizon).shift(-horizon)
        df['sp500_return_forward'] = df['sp500_price'].pct_change(periods=horizon).shift(-horizon)
        
        # Target = 1 if gold outperforms, 0 otherwise
        df['target'] = (df['gold_return_forward'] > df['sp500_return_forward']).astype(int)
        
        # Drop intermediate columns
        df = df.drop(columns=['gold_return_forward', 'sp500_return_forward'])
        
    elif method == 'price_increase':
        # Simpler: will gold price increase?
        df['gold_return_forward'] = df['gold_price'].pct_change(periods=horizon).shift(-horizon)
        df['target'] = (df['gold_return_forward'] > 0).astype(int)
        df = df.drop(columns=['gold_return_forward'])
    
    return df


def save_processed_data(df, output_dir='../data/processed', filename='gold_features.csv'):
    """
    Save processed dataset to CSV.
    
    Args:
        df: DataFrame to save
        output_dir: Output directory
        filename: Output filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / filename
    df.to_csv(filepath, index=False)
    
    print(f"\nSaved to: {filepath}")


def build_features(data_dir='../data/raw', 
                   start_date='2006-01-03',
                   end_date='2024-04-11',
                   target_horizon=90):
    """
    Complete pipeline: load, merge, engineer features, create target.
    
    Args:
        data_dir: Directory containing raw CSV files
        start_date: Start date for dataset
        end_date: End date for dataset
        target_horizon: Days forward for target calculation
        
    Returns:
        DataFrame with all features and target
    """
    
    # === STEP 1: Load all DAILY market data ===
    print("Loading daily market data...")
    
    gold = load_daily_data(f'{data_dir}/Gold Futures Historical Data.csv')
    sp500 = load_daily_data(f'{data_dir}/S&P 500 Historical Data.csv')
    silver = load_daily_data(f'{data_dir}/Silver Futures Historical Data.csv')
    usd = load_daily_data(f'{data_dir}/USD Index.csv')
    oil = load_daily_data(f'{data_dir}/Spot_Crude_Oil_Price.csv')
    
    # FIXED: These are daily FRED data, not monthly
    treasury = load_daily_data(f'{data_dir}/US_Treasury_Yield.csv')
    nasdaq = load_daily_data(f'{data_dir}/NASDAQCOM.csv')
    vix = load_daily_data(f'{data_dir}/VIXCLS.csv')
    
    print(f"  Gold: {len(gold)} rows")
    print(f"  S&P 500: {len(sp500)} rows")
    print(f"  NASDAQ: {len(nasdaq)} rows")
    print(f"  VIX: {len(vix)} rows")
    
    # === STEP 2: Standardize column names ===
    gold = standardize_columns(gold, 'gold')
    sp500 = standardize_columns(sp500, 'sp500')
    silver = standardize_columns(silver, 'silver')
    usd = standardize_columns(usd, 'usd_index')
    oil = standardize_columns(oil, 'oil')
    treasury = standardize_columns(treasury, 'treasury')
    nasdaq = standardize_columns(nasdaq, 'nasdaq')
    vix = standardize_columns(vix, 'vix')
    
    print(f"\nColumn names after standardization:")
    print(f"  Treasury: {[c for c in treasury.columns if c != 'date']}")
    print(f"  NASDAQ: {[c for c in nasdaq.columns if c != 'date']}")
    print(f"  VIX: {[c for c in vix.columns if c != 'date']}")
    
    # === STEP 3: Merge daily data (SMART STRATEGY) ===
    print("\nMerging daily datasets...")
    
    # Get actual column names (they vary by file)
    gold_col = [c for c in gold.columns if c != 'date'][0]
    sp500_col = [c for c in sp500.columns if c != 'date'][0]
    silver_col = [c for c in silver.columns if c != 'date'][0]
    usd_col = [c for c in usd.columns if c != 'date'][0]
    treasury_col = [c for c in treasury.columns if c != 'date'][0]
    nasdaq_col = [c for c in nasdaq.columns if c != 'date'][0]
    vix_col = [c for c in vix.columns if c != 'date'][0]
    oil_col = [c for c in oil.columns if c != 'date'][0]
    
    # CORE ASSETS: Inner join (must have all these)
    print("  Merging core assets (gold, S&P, silver)...")
    df = gold[['date', gold_col]].rename(columns={gold_col: 'gold_price'})
    
    temp = sp500[['date', sp500_col]].rename(columns={sp500_col: 'sp500_price'})
    df = df.merge(temp, on='date', how='inner')
    
    temp = silver[['date', silver_col]].rename(columns={silver_col: 'silver_price'})
    df = df.merge(temp, on='date', how='inner')
    
    print(f"    After core merge: {len(df)} rows")
    
    # SUPPLEMENTARY DATA: Left join (keep all core dates, fill NaN if missing)
    print("  Merging supplementary data...")
    
    temp = usd[['date', usd_col]].rename(columns={usd_col: 'usd_index_value'})
    df = df.merge(temp, on='date', how='left')
    
    temp = treasury[['date', treasury_col]].rename(columns={treasury_col: 'treasury_yield'})
    df = df.merge(temp, on='date', how='left')
    
    temp = nasdaq[['date', nasdaq_col]].rename(columns={nasdaq_col: 'nasdaq_value'})
    df = df.merge(temp, on='date', how='left')
    
    temp = vix[['date', vix_col]].rename(columns={vix_col: 'vix_value'})
    df = df.merge(temp, on='date', how='left')
    
    temp = oil[['date', oil_col]].rename(columns={oil_col: 'oil_price'})
    df = df.merge(temp, on='date', how='left')
    
    # Forward-fill any NaN values from supplementary data
    for col in ['usd_index_value', 'treasury_yield', 'nasdaq_value', 'vix_value', 'oil_price']:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()  # Fill forward, then backward for early dates
    
    print(f"  After merge: {len(df)} rows")
    
    # === STEP 4: Filter to target date range BEFORE rolling windows ===
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    print(f"  After date filter ({start_date} to {end_date}): {len(df)} rows")
    
    # === STEP 5: Engineer features on CLEAN daily data ===
    print("\nEngineering features...")
    df = create_returns(df, periods=[5, 20, 60])
    df = create_moving_averages(df, windows=[20])  # Only MA_20 to reduce NaNs
    df = create_volatility(df, windows=[20, 60])
    
    print(f"  After feature engineering: {len(df.columns)} columns")
    
    # === STEP 6: Load MONTHLY macro data ===
    print("\nLoading monthly macro data...")
    cpi = load_monthly_data(f'{data_dir}/CPI_1947.csv')
    unemployment = load_monthly_data(f'{data_dir}/unemployment.csv')
    fedfunds = load_monthly_data(f'{data_dir}/FEDFUNDS.csv')
    m2 = load_monthly_data(f'{data_dir}/M2_Money_Supply.csv')
    
    # Standardize
    cpi = standardize_columns(cpi, 'cpi')
    unemployment = standardize_columns(unemployment, 'unemployment')
    fedfunds = standardize_columns(fedfunds, 'fedfunds')
    m2 = standardize_columns(m2, 'm2')
    
    # === STEP 7: Merge monthly data with forward-fill ===
    print("Merging monthly data...")
    df = forward_fill_monthly(df, cpi, 'cpi')
    df = forward_fill_monthly(df, unemployment, 'unemployment')
    df = forward_fill_monthly(df, fedfunds, 'fedfunds')
    df = forward_fill_monthly(df, m2, 'm2')
    
    # === STEP 8: Create derived macro features ===
    print("Creating derived macro features...")
    df = create_macro_features(df)
    
    # === STEP 9: Create target variable ===
    print("Creating target variable...")
    df = create_target(df, horizon=target_horizon, method='outperform_sp500')
    
    # === STEP 10: Final cleanup ===
    df = df.dropna(subset=['target'])  # Remove rows without valid target
    

    print(f"\nPIPELINE COMPLETE")
    print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Target distribution: {df['target'].mean()*100:.1f}% positive")
    
    # Check for remaining NaNs
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) > 0:
        print(f"\nFeatures with missing values:")
        for col, count in missing.head(5).items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print("\nNo missing values!")
    
    return df