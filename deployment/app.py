"""
Simple Streamlit Frontend for Gold Prediction
"""
import streamlit as st
import requests
import pandas as pd

# Page config
st.set_page_config(page_title="Gold vs Stocks Predictor", page_icon="💰")

# Title
st.title("💰 Gold vs Stocks Predictor")
st.write("Will gold outperform stocks in the next 90 days?")

# Tabs for individual vs batch
tab1, tab2 = st.tabs(["🔮 Single Prediction", "📊 Batch Predictions"])

# TAB 1: SINGLE PREDICTION
with tab1:
    st.header("Enter Market Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gold_price = st.number_input("Gold Price ($)", value=1850.0)
        sp500_price = st.number_input("S&P 500", value=4200.0)
        silver_price = st.number_input("Silver Price ($)", value=23.5)
        usd_index = st.number_input("USD Index", value=103.0)
    
    with col2:
        treasury = st.number_input("Treasury Yield (%)", value=4.5)
        nasdaq = st.number_input("NASDAQ", value=13500.0)
        vix = st.number_input("VIX", value=15.0)
        oil = st.number_input("Oil Price ($)", value=78.0)
    
    # Predict button
    if st.button("🔮 Make Prediction", type="primary"):
        # Prepare data
        data = {
            "gold_price": gold_price,
            "sp500_price": sp500_price,
            "silver_price": silver_price,
            "usd_index_value": usd_index,
            "treasury_yield": treasury,
            "nasdaq_value": nasdaq,
            "vix_value": vix,
            "oil_price": oil
        }
        
        # Call API
        try:
            response = requests.post("http://localhost:8000/predict", json=data)
            result = response.json()
            
            # Show result
            st.success("Prediction Complete!")
            st.subheader(result["winner"])
            st.metric("Probability", result["probability"])
            
        except Exception as e:
            st.error("Error: Make sure the API is running!")
            st.code("python api.py")

# TAB 2: BATCH PREDICTIONS
with tab2:
    st.header("Upload CSV for Batch Predictions")
    
    # Show sample format
    st.write("**Required columns:** gold_price, sp500_price, silver_price, usd_index_value, treasury_yield, nasdaq_value, vix_value, oil_price")
    
    # Sample CSV download
    sample = pd.DataFrame({
        'gold_price': [1850, 1900, 1800],
        'sp500_price': [4200, 4300, 4100],
        'silver_price': [23.5, 24.0, 23.0],
        'usd_index_value': [103, 102, 104],
        'treasury_yield': [4.5, 4.6, 4.4],
        'nasdaq_value': [13500, 13600, 13400],
        'vix_value': [15, 16, 14],
        'oil_price': [78, 80, 76]
    })
    st.download_button("📥 Download Sample CSV", sample.to_csv(index=False), "sample.csv")
    
    # Upload file
    uploaded = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"**Loaded {len(df)} rows**")
        st.dataframe(df.head())
        
        if st.button("🚀 Run Batch Prediction", type="primary"):
            try:
                results = []
                for _, row in df.iterrows():
                    data = {
                        "gold_price": row['gold_price'],
                        "sp500_price": row['sp500_price'],
                        "silver_price": row['silver_price'],
                        "usd_index_value": row['usd_index_value'],
                        "treasury_yield": row['treasury_yield'],
                        "nasdaq_value": row['nasdaq_value'],
                        "vix_value": row['vix_value'],
                        "oil_price": row['oil_price']
                    }
                    response = requests.post("http://localhost:8000/predict", json=data)
                    results.append(response.json())
                
                # Show results
                st.success(f"✅ Completed {len(results)} predictions!")
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Summary
                gold_wins = (results_df['prediction'] == 'Gold').sum()
                st.metric("Gold Wins", f"{gold_wins} / {len(results)}")
                
                # Download
                st.download_button(
                    "📥 Download Results",
                    results_df.to_csv(index=False),
                    "predictions.csv"
                )
                
            except Exception as e:
                st.error(f"Error: {e}")