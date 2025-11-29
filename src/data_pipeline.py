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


