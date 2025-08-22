"""
Test script for technical indicators module.
Tests RSI, MACD, Bollinger Bands and other indicators with real market data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.screening.technical_indicators import TechnicalAnalyzer, TechnicalIndicators
from src.data_pipeline.fetcher import StockDataFetcher
from src.data_pipeline.sp500_scraper import SP500Scraper
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict


def print_separator(char="-", length=60):
    """Print a separator line."""
    print(char * length)


def display_indicator_details(indicators: TechnicalIndicators):
    """Display detailed technical indicators for a stock."""
    print(f"\n{'='*60}")
    print(f"Technical Analysis: {indicators.ticker}")
    print(f"{'='*60}")
    
    print("\nMomentum Indicators:")
    print(f"  • RSI: {f'{indicators.rsi:.2f}' if indicators.rsi is not None else 'N/A'} ({indicators.rsi_signal if indicators.rsi_signal else 'N/A'})")
    print(f"  • MACD: {f'{indicators.macd:.4f}' if indicators.macd is not None else 'N/A'}")
    print(f"  • MACD Signal: {f'{indicators.macd_signal:.4f}' if indicators.macd_signal is not None else 'N/A'}")
    print(f"  • MACD Histogram: {f'{indicators.macd_histogram:.4f}' if indicators.macd_histogram is not None else 'N/A'}")
    print(f"  • MACD Crossover: {indicators.macd_crossover if indicators.macd_crossover else 'N/A'}")
    
    print("\nBollinger Bands:")
    print(f"  • Upper Band: ${f'{indicators.bb_upper:.2f}' if indicators.bb_upper is not None else 'N/A'}")
    print(f"  • Middle Band: ${f'{indicators.bb_middle:.2f}' if indicators.bb_middle is not None else 'N/A'}")
    print(f"  • Lower Band: ${f'{indicators.bb_lower:.2f}' if indicators.bb_lower is not None else 'N/A'}")
    print(f"  • BB Position: {f'{indicators.bb_percent:.2%}' if indicators.bb_percent is not None else 'N/A'}")
    print(f"  • BB Signal: {indicators.bb_signal if indicators.bb_signal else 'N/A'}")
    
    print("\nMoving Averages:")
    print(f"  • SMA 20: ${f'{indicators.sma_20:.2f}' if indicators.sma_20 is not None else 'N/A'}")
    print(f"  • SMA 50: ${f'{indicators.sma_50:.2f}' if indicators.sma_50 is not None else 'N/A'}")
    print(f"  • SMA 200: ${f'{indicators.sma_200:.2f}' if indicators.sma_200 is not None else 'N/A'}")
    print(f"  • EMA 12: ${f'{indicators.ema_12:.2f}' if indicators.ema_12 is not None else 'N/A'}")
    print(f"  • EMA 26: ${f'{indicators.ema_26:.2f}' if indicators.ema_26 is not None else 'N/A'}")
    
    print("\nVolume Analysis:")
    print(f"  • Volume Ratio: {f'{indicators.volume_ratio:.2f}x' if indicators.volume_ratio is not None else 'N/A'}")
    print(f"  • OBV: {f'{indicators.obv:,.0f}' if indicators.obv is not None else 'N/A'}")
    print(f"  • VWAP: ${f'{indicators.vwap:.2f}' if indicators.vwap is not None else 'N/A'}")
    
    print("\nTrend & Volatility:")
    trend_strength = "Strong Trend" if indicators.adx and indicators.adx > 25 else "Weak Trend" if indicators.adx else "N/A"
    print(f"  • ADX: {f'{indicators.adx:.2f}' if indicators.adx is not None else 'N/A'} ({trend_strength})")
    print(f"  • ATR: {f'{indicators.atr:.2f}' if indicators.atr is not None else 'N/A'}")
    print(f"  • Stochastic %K: {f'{indicators.stochastic_k:.2f}' if indicators.stochastic_k is not None else 'N/A'}")
    print(f"  • Stochastic %D: {f'{indicators.stochastic_d:.2f}' if indicators.stochastic_d is not None else 'N/A'}")


def test_single_stock(ticker: str = "AAPL"):
    """Test technical indicators for a single stock."""
    print(f"\n*** Testing Technical Indicators for {ticker} ***")
    
    # Fetch data
    fetcher = StockDataFetcher()
    print(f"Fetching data for {ticker}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Get 1 year of data for good indicators
    
    df = fetcher.fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if df.empty:
        print(f"❌ Failed to fetch data for {ticker}")
        return None
    
    print(f"✓ Fetched {len(df)} days of data")
    
    # Calculate indicators
    analyzer = TechnicalAnalyzer()
    indicators = analyzer.analyze_stock(df, ticker)
    
    # Display results
    display_indicator_details(indicators)
    
    # Get bullish signals
    bullish_signals = analyzer.get_bullish_signals(indicators)
    if bullish_signals:
        print(f"\n✅ Bullish Signals Found:")
        for signal in bullish_signals:
            print(f"  • {signal}")
    else:
        print("\n⚠️ No strong bullish signals detected")
    
    # Calculate technical score
    score = analyzer.score_stock(indicators)
    print(f"\nTechnical Score: {score:.1f}/100 {'📈' if score >= 70 else '➡️' if score >= 40 else '📉'}")
    
    return indicators


def test_multiple_stocks():
    """Test technical indicators for multiple S&P 500 stocks."""
    print("\n*** Testing Technical Indicators for Multiple Stocks ***")
    
    # Get some S&P 500 stocks
    scraper = SP500Scraper()
    tickers = scraper.fetch_sp500_tickers()[:10]  # Test with first 10 stocks
    
    print(f"Testing indicators for: {', '.join(tickers)}")
    
    # Fetch data for all stocks
    fetcher = StockDataFetcher()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    stock_data = {}
    print("Fetching stock data...")
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] Fetching {ticker}...", end=" ")
        df = fetcher.fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if not df.empty:
            stock_data[ticker] = df
            print("✓")
        else:
            print("✗")
    
    print(f"\n✓ Fetched data for {len(stock_data)} stocks")
    
    # Analyze all stocks
    analyzer = TechnicalAnalyzer()
    print("\nAnalyzing stocks...")
    results = analyzer.analyze_multiple_stocks(stock_data, max_workers=5)
    
    # Create results table
    print("\n" + "="*80)
    print("Technical Analysis Results")
    print("="*80)
    print(f"{'Ticker':<8} {'RSI':<8} {'Signal':<12} {'MACD':<10} {'BB Pos':<10} {'Vol Ratio':<10} {'ADX':<8} {'Score':<8}")
    print("-"*80)
    
    # Sort by score
    sorted_results = sorted(results.items(), 
                           key=lambda x: analyzer.score_stock(x[1]), 
                           reverse=True)
    
    for ticker, indicators in sorted_results:
        score = analyzer.score_stock(indicators)
        
        rsi_str = f"{indicators.rsi:.1f}" if indicators.rsi else "N/A"
        rsi_signal = indicators.rsi_signal or "N/A"
        macd_cross = indicators.macd_crossover or "N/A"
        bb_pos = f"{indicators.bb_percent:.1%}" if indicators.bb_percent else "N/A"
        vol_ratio = f"{indicators.volume_ratio:.1f}x" if indicators.volume_ratio else "N/A"
        adx_str = f"{indicators.adx:.1f}" if indicators.adx else "N/A"
        
        print(f"{ticker:<8} {rsi_str:<8} {rsi_signal:<12} {macd_cross:<10} {bb_pos:<10} {vol_ratio:<10} {adx_str:<8} {score:<8.1f}")
    
    # Find best opportunities
    print("\n✨ Top Technical Opportunities:")
    for ticker, indicators in sorted_results[:3]:
        signals = analyzer.get_bullish_signals(indicators)
        score = analyzer.score_stock(indicators)
        if signals:
            print(f"  {ticker} (Score: {score:.1f}): {', '.join(signals)}")


def test_indicator_accuracy():
    """Test the accuracy of indicator calculations with known values."""
    print("\n*** Testing Indicator Calculation Accuracy ***")
    
    # Create sample data with known values
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    
    # Create predictable price pattern for testing
    prices = [100 + i * 0.5 + (5 * (i % 10 - 5)) for i in range(50)]
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p + 2 for p in prices],
        'low': [p - 2 for p in prices],
        'close': prices,
        'volume': [1000000 + i * 10000 for i in range(50)]
    })
    
    analyzer = TechnicalAnalyzer()
    
    # Test RSI
    print("\nTesting RSI calculation...")
    rsi = analyzer.calculate_rsi(df)
    print(f"  Last RSI value: {rsi.iloc[-1]:.2f}")
    print(f"  RSI range: {rsi.min():.2f} - {rsi.max():.2f}")
    assert 0 <= rsi.iloc[-1] <= 100, "RSI should be between 0 and 100"
    print("  ✓ RSI calculation passed")
    
    # Test MACD
    print("\nTesting MACD calculation...")
    macd = analyzer.calculate_macd(df)
    print(f"  MACD columns: {list(macd.columns)}")
    print(f"  Last MACD value: {macd.iloc[-1, 0]:.4f}")
    assert not macd.empty, "MACD should return values"
    print("  ✓ MACD calculation passed")
    
    # Test Bollinger Bands
    print("\nTesting Bollinger Bands calculation...")
    bbands = analyzer.calculate_bollinger_bands(df)
    print(f"  BB columns: {list(bbands.columns)}")
    upper = bbands.iloc[-1, 2]
    middle = bbands.iloc[-1, 1]
    lower = bbands.iloc[-1, 0]
    print(f"  Upper: {upper:.2f}, Middle: {middle:.2f}, Lower: {lower:.2f}")
    assert lower < middle < upper, "Bollinger Bands should be ordered: lower < middle < upper"
    print("  ✓ Bollinger Bands calculation passed")
    
    # Test Moving Averages
    print("\nTesting Moving Averages calculation...")
    mas = analyzer.calculate_moving_averages(df)
    sma20 = mas['sma_20'].iloc[-1]
    print(f"  SMA(20): {sma20:.2f}")
    expected_sma20 = df['close'].iloc[-20:].mean()
    assert abs(sma20 - expected_sma20) < 0.01, "SMA calculation mismatch"
    print("  ✓ Moving Averages calculation passed")
    
    print("\n✅ All indicator calculations passed!")


def test_breakout_detection():
    """Test detection of breakout patterns using technical indicators."""
    print("\n*** Testing Breakout Pattern Detection ***")
    
    # Get stocks that recently had significant moves
    scraper = SP500Scraper()
    fetcher = StockDataFetcher()
    analyzer = TechnicalAnalyzer()
    
    # Fetch some volatile stocks
    test_tickers = ['NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT']
    print(f"Analyzing potential breakouts in: {', '.join(test_tickers)}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    breakout_candidates = []
    
    for ticker in test_tickers:
        print(f"  Analyzing {ticker}...", end=" ")
        df = fetcher.fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if df.empty:
            print("✗ No data")
            continue
        
        indicators = analyzer.analyze_stock(df, ticker)
        score = analyzer.score_stock(indicators)
        signals = analyzer.get_bullish_signals(indicators)
        
        # Check for breakout conditions
        breakout_conditions = []
        
        # RSI recovering from oversold
        if indicators.rsi and 30 < indicators.rsi < 50:
            breakout_conditions.append("RSI_RECOVERY")
        
        # MACD bullish crossover
        if indicators.macd_crossover == "bullish":
            breakout_conditions.append("MACD_BULLISH")
        
        # Bollinger Band squeeze
        if indicators.bb_signal == "squeeze":
            breakout_conditions.append("BB_SQUEEZE")
        
        # Volume surge
        if indicators.volume_ratio and indicators.volume_ratio > 1.5:
            breakout_conditions.append("VOLUME_SURGE")
        
        if breakout_conditions:
            breakout_candidates.append({
                'ticker': ticker,
                'score': score,
                'conditions': breakout_conditions,
                'signals': signals
            })
            print(f"✓ Score: {score:.1f}")
        else:
            print(f"○ Score: {score:.1f}")
    
    if breakout_candidates:
        print("\n🎯 Potential Breakout Candidates Found:")
        for candidate in sorted(breakout_candidates, key=lambda x: x['score'], reverse=True):
            print(f"\n  {candidate['ticker']} (Score: {candidate['score']:.1f})")
            print(f"    Conditions: {', '.join(candidate['conditions'])}")
            print(f"    Signals: {', '.join(candidate['signals'])}")
    else:
        print("\n⚠️ No strong breakout candidates found in test set")


def main():
    """Run all technical indicator tests."""
    print("\n" + "="*60)
    print("TECHNICAL INDICATORS TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Single stock analysis
        print("\n[Test 1: Single Stock Analysis]")
        print_separator()
        test_single_stock("AAPL")
        
        time.sleep(1)  # Small delay between tests
        
        # Test 2: Indicator accuracy
        print("\n[Test 2: Indicator Calculation Accuracy]")
        print_separator()
        test_indicator_accuracy()
        
        time.sleep(1)
        
        # Test 3: Multiple stocks
        print("\n[Test 3: Multiple Stock Analysis]")
        print_separator()
        test_multiple_stocks()
        
        time.sleep(1)
        
        # Test 4: Breakout detection
        print("\n[Test 4: Breakout Pattern Detection]")
        print_separator()
        test_breakout_detection()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"{str(e)}")
        import traceback
        print(traceback.format_exc())
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())