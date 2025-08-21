"""
Test single stock data fetching to diagnose issues
"""

import sys
import os
sys.path.append('src')

from data_pipeline.fetcher import StockDataFetcher
import yfinance as yf
import time

def test_yfinance_directly():
    """Test yfinance directly to isolate the issue"""
    print("="*60)
    print("TESTING YFINANCE DIRECTLY")
    print("="*60)
    
    # Test with known good tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    for ticker in test_tickers:
        print(f"\nTesting {ticker}...")
        try:
            # Method 1: Using Ticker object
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            
            if not hist.empty:
                print(f"✓ {ticker}: Got {len(hist)} days of data")
                print(f"  Latest close: ${hist['Close'].iloc[-1]:.2f}")
            else:
                print(f"✗ {ticker}: No data returned")
                
            # Small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"✗ {ticker}: Error - {e}")
    
    # Test batch download
    print("\n" + "-"*40)
    print("Testing batch download...")
    try:
        data = yf.download(
            tickers=" ".join(test_tickers),
            period="5d",
            progress=False,
            threads=False  # Disable threading to avoid issues
        )
        
        if not data.empty:
            print(f"✓ Batch download successful: {data.shape}")
        else:
            print("✗ Batch download returned empty data")
            
    except Exception as e:
        print(f"✗ Batch download error: {e}")

def test_with_our_fetcher():
    """Test using our fetcher class"""
    print("\n" + "="*60)
    print("TESTING OUR FETCHER")
    print("="*60)
    
    fetcher = StockDataFetcher()
    
    # Test single stock with reduced rate
    fetcher.requests_per_second = 0.5  # Slower rate
    
    test_ticker = "AAPL"
    print(f"\nFetching {test_ticker} with our fetcher...")
    
    df = fetcher.fetch_stock_data(
        test_ticker,
        period="5d",
        use_cache=False  # Force fresh fetch
    )
    
    if not df.empty:
        print(f"✓ Got {len(df)} days of data")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Latest close: ${df['Close'].iloc[-1]:.2f}")
    else:
        print("✗ No data returned")

def diagnose_connection():
    """Check if we can connect to Yahoo Finance"""
    print("\n" + "="*60)
    print("DIAGNOSING CONNECTION")
    print("="*60)
    
    import requests
    
    # Test basic connectivity
    urls_to_test = [
        "https://www.google.com",
        "https://finance.yahoo.com",
        "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
    ]
    
    for url in urls_to_test:
        try:
            response = requests.get(url, timeout=5)
            print(f"✓ {url}: Status {response.status_code}")
        except Exception as e:
            print(f"✗ {url}: {e}")

if __name__ == "__main__":
    # First check connectivity
    diagnose_connection()
    
    # Test yfinance directly
    test_yfinance_directly()
    
    # Test our fetcher
    test_with_our_fetcher()
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    print("\nIf all tests fail, possible solutions:")
    print("1. Update yfinance: pip install --upgrade yfinance")
    print("2. Check firewall/proxy settings")
    print("3. Try using a VPN")
    print("4. Wait and retry (rate limiting)")