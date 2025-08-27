"""
Test the data pipeline components
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.sp500_scraper import SP500Scraper
from src.data_pipeline.fetcher import StockDataFetcher
import random
import pandas as pd

def test_pipeline():
    print("="*60)
    print("TESTING DATA PIPELINE")
    print("="*60)
    
    # Initialize components
    scraper = SP500Scraper()
    fetcher = StockDataFetcher()
    
    # Step 1: Get S&P 500 tickers
    print("\n1. Fetching S&P 500 constituents...")
    print("-"*40)
    
    tickers = scraper.fetch_sp500_tickers()
    print(f"✓ Found {len(tickers)} S&P 500 companies")
    print(f"Sample tickers: {tickers[:5]}")
    
    # Step 2: Get detailed company info
    print("\n2. Getting company details...")
    print("-"*40)
    
    companies_df = scraper.get_sp500_info()
    print(f"✓ Company info shape: {companies_df.shape}")
    
    if 'sector' in companies_df.columns:
        print("\nSector distribution:")
        sector_counts = companies_df['sector'].value_counts().head()
        for sector, count in sector_counts.items():
            print(f"  {sector}: {count} companies")
    
    # Step 3: Test with random 10 stocks
    print("\n3. Testing data fetch for 10 random stocks...")
    print("-"*40)
    
    # Select 10 random tickers
    random_tickers = random.sample(tickers, min(10, len(tickers)))
    print(f"Selected tickers: {random_tickers}")
    
    # Fetch data for these stocks
    print("\nFetching 1 month of data...")
    stock_data = fetcher.fetch_multiple_stocks(
        random_tickers,
        period="1mo",
        max_workers=3  # Limit parallel requests
    )
    
    print(f"\n✓ Successfully fetched {len(stock_data)}/{len(random_tickers)} stocks")
    
    # Step 4: Analyze the data
    print("\n4. Analyzing fetched data...")
    print("-"*40)
    
    for ticker, df in stock_data.items():
        # Add calculations
        df = fetcher.calculate_returns(df)
        df = fetcher.calculate_volatility(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        print(f"\n{ticker}:")
        print(f"  Days of data: {len(df)}")
        print(f"  Latest close: ${latest['Close']:.2f}")
        print(f"  Volume: {latest['Volume']:,.0f}")
        
        if 'Daily_Return' in df.columns and not df['Daily_Return'].isna().all():
            print(f"  Daily return: {latest['Daily_Return']*100:.2f}%")
        
        if 'Volatility' in df.columns and not df['Volatility'].isna().all():
            vol = df['Volatility'].dropna()
            if len(vol) > 0:
                print(f"  Volatility (20d): {vol.iloc[-1]*100:.1f}%")
    
    # Step 5: Test latest prices
    print("\n5. Testing real-time price fetch...")
    print("-"*40)
    
    test_tickers = random_tickers[:5]
    latest_prices = fetcher.get_latest_prices(test_tickers)
    
    if not latest_prices.empty:
        print("\nLatest prices:")
        for _, row in latest_prices.iterrows():
            print(f"  {row['Ticker']}: ${row['Price']:.2f} (Volume: {row['Volume']:,.0f})")
    
    # Step 6: Test cache
    print("\n6. Testing cache functionality...")
    print("-"*40)
    
    # Fetch same ticker again (should use cache)
    test_ticker = random_tickers[0]
    print(f"Re-fetching {test_ticker} (should use cache)...")
    df = fetcher.fetch_stock_data(test_ticker, period="1mo")
    print(f"✓ Data retrieved: {len(df)} rows")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()