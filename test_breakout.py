"""
Test the consolidated breakout filter
"""

import sys
sys.path.append('src')

from screening.breakout_filter import find_breakouts, FilterCriteria, BreakoutFilter
from data_pipeline.fetcher import StockDataFetcher
from data_pipeline.sp500_scraper import SP500Scraper

def test_filter_modes():
    """Test different filter modes"""
    print("="*60)
    print("TESTING BREAKOUT FILTER - ALL MODES")
    print("="*60)
    
    # Test with 30 stocks for speed
    test_limit = 30
    
    modes = ["strict", "normal", "loose"]
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Testing {mode.upper()} mode with {test_limit} stocks")
        print("-"*40)
        
        results = find_breakouts(
            mode=mode,
            ticker_limit=test_limit,
            max_results=3  # Show top 3 only
        )
        
        if results:
            print(f"\nPass rate: {len(results)}/{test_limit} stocks")
        else:
            print("\nNo stocks passed (this is fine for strict mode!)")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

def test_specific_stocks():
    """Test with specific known stocks"""
    print("\n" + "="*60)
    print("TESTING SPECIFIC STOCKS")
    print("="*60)
    
    # Test with big tech stocks
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
    
    fetcher = StockDataFetcher()
    criteria = FilterCriteria.normal()  # Use normal mode
    filter = BreakoutFilter(fetcher, criteria)
    
    print(f"\nAnalyzing: {', '.join(test_tickers)}")
    print("-"*40)
    
    results = filter.analyze_stocks(test_tickers, max_results=10)
    
    if results:
        filter.print_results(results)
    else:
        print("\nNone of these stocks are in breakout mode currently")
        print("(This is normal - breakouts are rare!)")

def test_criteria_comparison():
    """Compare different criteria settings"""
    print("\n" + "="*60)
    print("COMPARING FILTER CRITERIA")
    print("="*60)
    
    # Show criteria differences
    strict = FilterCriteria.ultra_strict()
    normal = FilterCriteria.normal()
    loose = FilterCriteria.loose()
    
    print("\n📊 Criteria Comparison:")
    print("-"*40)
    print(f"{'Setting':<25} {'Strict':<15} {'Normal':<15} {'Loose':<15}")
    print("-"*40)
    print(f"{'Min Price':<25} ${strict.min_price:<14.0f} ${normal.min_price:<14.0f} ${loose.min_price:<14.0f}")
    print(f"{'Min Volume':<25} {strict.min_volume:,<15} {normal.min_volume:,<15} {loose.min_volume:,<15}")
    print(f"{'Price Change Range':<25} {strict.min_price_change*100:.0f}-{strict.max_price_change*100:.0f}%{'':<10} "
          f"{normal.min_price_change*100:.0f}-{normal.max_price_change*100:.0f}%{'':<10} "
          f"{loose.min_price_change*100:.0f}-{loose.max_price_change*100:.0f}%")
    print(f"{'Min Momentum':<25} {strict.min_momentum*100:.0f}%{'':<14} {normal.min_momentum*100:.0f}%{'':<14} {loose.min_momentum*100:.0f}%")
    print(f"{'Volume Ratio Required':<25} {strict.min_volume_ratio:.1f}x{'':<13} {normal.min_volume_ratio:.1f}x{'':<13} {loose.min_volume_ratio:.1f}x")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test breakout filter")
    parser.add_argument("--test", choices=["all", "modes", "stocks", "criteria"],
                       default="modes", help="Which test to run")
    
    args = parser.parse_args()
    
    if args.test == "all":
        test_filter_modes()
        test_specific_stocks()
        test_criteria_comparison()
    elif args.test == "modes":
        test_filter_modes()
    elif args.test == "stocks":
        test_specific_stocks()
    elif args.test == "criteria":
        test_criteria_comparison()