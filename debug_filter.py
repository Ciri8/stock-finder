"""
Debug the filter to understand why so many stocks are passing
"""

import sys
sys.path.append('src')

from screening.initial_filter import InitialStockFilter, FilterCriteria
from data_pipeline.fetcher import StockDataFetcher
from data_pipeline.sp500_scraper import SP500Scraper

def main():
    print("="*60)
    print("FILTER DEBUG - Understanding Pass/Fail Reasons")
    print("="*60)
    
    # Initialize
    fetcher = StockDataFetcher()
    scraper = SP500Scraper()
    
    # Use current production criteria
    criteria = FilterCriteria(
        min_volume=2_000_000,
        min_price_change=0.045,
        max_price_change=0.075,
        min_price=30.0,
        max_price=1000.0,
        lookback_days=5,
        volume_lookback_days=20,
    )
    
    filter = InitialStockFilter(fetcher, criteria)
    
    # Get first 20 stocks for detailed analysis
    tickers = scraper.fetch_sp500_tickers()[:20]
    
    print(f"\nAnalyzing {len(tickers)} stocks in detail...")
    print("-"*60)
    
    # Get the data
    stock_data = fetcher.fetch_multiple_stocks(tickers, period="20d")
    
    # Manually check each stock
    passed_count = 0
    failed_count = 0
    
    print("\n📊 DETAILED ANALYSIS:")
    print("-"*60)
    
    for ticker, df in stock_data.items():
        if len(df) < 5:
            print(f"\n{ticker}: ❌ Insufficient data")
            failed_count += 1
            continue
            
        # Calculate metrics
        current_price = df['Close'].iloc[-1]
        week_ago_price = df['Close'].iloc[-5]
        weekly_change = (current_price - week_ago_price) / week_ago_price
        avg_volume = df['Volume'].mean()
        
        # Check each criterion
        checks = {
            'price_range': criteria.min_price <= current_price <= criteria.max_price,
            'weekly_change': criteria.min_price_change <= weekly_change <= criteria.max_price_change,
            'volume': avg_volume >= criteria.min_volume
        }
        
        passed_all = all(checks.values())
        
        if passed_all:
            passed_count += 1
            status = "✅ PASSED"
        else:
            failed_count += 1
            status = "❌ FAILED"
        
        print(f"\n{ticker}: {status}")
        print(f"  Price: ${current_price:.2f}", 
              "✓" if checks['price_range'] else f"✗ (need ${criteria.min_price}-${criteria.max_price})")
        print(f"  Weekly: {weekly_change*100:+.1f}%",
              "✓" if checks['weekly_change'] else f"✗ (need {criteria.min_price_change*100:.1f}%-{criteria.max_price_change*100:.1f}%)")
        print(f"  Volume: {avg_volume:,.0f}",
              "✓" if checks['volume'] else f"✗ (need {criteria.min_volume:,})")
        
        # Show why it passed/failed
        if not passed_all:
            failed_checks = [k for k, v in checks.items() if not v]
            print(f"  Failed: {', '.join(failed_checks)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Passed: {passed_count}/{len(stock_data)} ({passed_count/len(stock_data)*100:.1f}%)")
    print(f"  Failed: {failed_count}/{len(stock_data)}")
    
    print("\n💡 INSIGHTS:")
    if passed_count/len(stock_data) > 0.5:
        print("  ⚠️ Too many stocks passing!")
        print("  • Tighten price change range (try 5-6.5%)")
        print("  • Increase minimum volume (try 5M)")
        print("  • Increase minimum price (try $50)")
    else:
        print("  ✅ Filter is appropriately selective")
    
    print("="*60)

if __name__ == "__main__":
    main()