"""
Data Fetcher Module
Downloads and caches stock market data using yfinance
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetches and manages stock market data"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the data fetcher
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        self.price_cache_dir = os.path.join(cache_dir, "prices")
        
        # Create cache directories
        os.makedirs(self.price_cache_dir, exist_ok=True)
        
        # Rate limiting settings
        self.requests_per_second = 2  # Be nice to Yahoo
        self.last_request_time = 0
        
    def fetch_stock_data(self, 
                        ticker: str,
                        start_date: str = None,
                        end_date: str = None,
                        period: str = "1y",
                        interval: str = "1d",
                        use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single stock
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate cache key
        cache_key = self._get_cache_key(ticker, start_date, end_date, period, interval)
        cache_file = os.path.join(self.price_cache_dir, f"{cache_key}.pkl")
        
        # Check cache first
        if use_cache and os.path.exists(cache_file):
            cache_age = self._get_file_age_hours(cache_file)
            
            # Use cache if less than 24 hours old for daily data
            if interval == "1d" and cache_age < 24:
                logger.info(f"Loading {ticker} from cache (age: {cache_age:.1f} hours)")
                return pd.read_pickle(cache_file)
            # Use cache if less than 1 hour old for intraday data
            elif interval != "1d" and cache_age < 1:
                logger.info(f"Loading {ticker} from cache (age: {cache_age:.1f} hours)")
                return pd.read_pickle(cache_file)
        
        # Rate limiting
        self._rate_limit()
        
        # Fetch from yfinance
        logger.info(f"Fetching {ticker} from Yahoo Finance")
        try:
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date, interval=interval)
            else:
                df = stock.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()
            
            # Add ticker column
            df['Ticker'] = ticker
            
            # Save to cache
            df.to_pickle(cache_file)
            logger.info(f"Cached {ticker} data ({len(df)} rows)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            
            # Try to return cached data even if expired
            if os.path.exists(cache_file):
                logger.info(f"Returning expired cache for {ticker} due to fetch error")
                return pd.read_pickle(cache_file)
            
            return pd.DataFrame()
    
    def fetch_multiple_stocks(self,
                            tickers: List[str],
                            start_date: str = None,
                            end_date: str = None,
                            period: str = "1y",
                            interval: str = "1d",
                            max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks in parallel
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            period: Period to download
            interval: Data interval
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        results = {}
        
        logger.info(f"Fetching data for {len(tickers)} stocks with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(
                    self.fetch_stock_data,
                    ticker,
                    start_date,
                    end_date,
                    period,
                    interval
                ): ticker
                for ticker in tickers
            }
            
            # Collect results
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if not df.empty:
                        results[ticker] = df
                        logger.info(f" {ticker}: {len(df)} rows")
                    else:
                        logger.warning(f" {ticker}: No data")
                except Exception as e:
                    logger.error(f" {ticker}: {e}")
        
        logger.info(f"Successfully fetched {len(results)}/{len(tickers)} stocks")
        return results
    
    def get_latest_prices(self, tickers: List[str]) -> pd.DataFrame:
        """
        Get latest price for multiple stocks (real-time during market hours)
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with latest prices
        """
        logger.info(f"Fetching latest prices for {len(tickers)} stocks")
        
        # Rate limiting
        self._rate_limit()
        
        try:
            # Download all at once (more efficient)
            data = yf.download(
                tickers=" ".join(tickers),
                period="1d",
                interval="1d",
                progress=False,
                threads=True
            )
            
            if len(tickers) == 1:
                # Single ticker returns different structure
                latest = pd.DataFrame([{
                    'Ticker': tickers[0],
                    'Price': data['Close'].iloc[-1],
                    'Volume': data['Volume'].iloc[-1],
                    'Date': data.index[-1]
                }])
            else:
                # Multiple tickers
                latest_data = []
                for ticker in tickers:
                    try:
                        latest_data.append({
                            'Ticker': ticker,
                            'Price': data['Close'][ticker].iloc[-1],
                            'Volume': data['Volume'][ticker].iloc[-1],
                            'Date': data.index[-1]
                        })
                    except:
                        logger.warning(f"Could not get latest price for {ticker}")
                
                latest = pd.DataFrame(latest_data)
            
            return latest
            
        except Exception as e:
            logger.error(f"Error fetching latest prices: {e}")
            return pd.DataFrame()
    
    def get_stock_info(self, ticker: str) -> Dict:
        """
        Get detailed stock information (company info, metrics, etc.)
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with stock information
        """
        logger.info(f"Fetching info for {ticker}")
        
        # Rate limiting
        self._rate_limit()
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key metrics
            return {
                'ticker': ticker,
                'name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', None),
                '52w_high': info.get('fiftyTwoWeekHigh', None),
                '52w_low': info.get('fiftyTwoWeekLow', None),
                'avg_volume': info.get('averageVolume', 0),
                'shares_outstanding': info.get('sharesOutstanding', 0)
            }
            
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return {'ticker': ticker, 'error': str(e)}
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return metrics
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added return columns
        """
        df = df.copy()
        
        # Daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Log returns (better for statistical properties)
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Cumulative returns
        df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
        
        # Moving average returns
        df['MA5_Return'] = df['Close'].rolling(5).mean().pct_change()
        df['MA20_Return'] = df['Close'].rolling(20).mean().pct_change()
        
        return df
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Calculate volatility metrics
        
        Args:
            df: DataFrame with OHLCV data
            window: Rolling window for volatility calculation
            
        Returns:
            DataFrame with added volatility columns
        """
        df = df.copy()
        
        # Historical volatility (annualized)
        df['Volatility'] = df['Daily_Return'].rolling(window).std() * np.sqrt(252)
        
        # Parkinson volatility (using high-low)
        df['Parkinson_Vol'] = np.sqrt(
            252 / (4 * np.log(2)) * 
            df['High'].div(df['Low']).apply(np.log).pow(2).rolling(window).mean()
        )
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['ATR'] = ranges.max(axis=1).rolling(window).mean()
        
        return df
    
    def _get_cache_key(self, ticker: str, start: str, end: str, period: str, interval: str) -> str:
        """Generate cache key for data"""
        key_parts = [ticker, interval]
        
        if start and end:
            key_parts.extend([start, end])
        else:
            key_parts.append(period)
        
        return "_".join(key_parts).replace("-", "")
    
    def _get_file_age_hours(self, filepath: str) -> float:
        """Get age of file in hours"""
        if not os.path.exists(filepath):
            return float('inf')
        
        file_time = os.path.getmtime(filepath)
        age_seconds = time.time() - file_time
        return age_seconds / 3600
    
    def _rate_limit(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def clear_cache(self, older_than_hours: int = None):
        """
        Clear cached data
        
        Args:
            older_than_hours: Only clear files older than this many hours
        """
        cleared = 0
        
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                filepath = os.path.join(root, file)
                
                if older_than_hours:
                    age = self._get_file_age_hours(filepath)
                    if age > older_than_hours:
                        os.remove(filepath)
                        cleared += 1
                else:
                    os.remove(filepath)
                    cleared += 1
        
        logger.info(f"Cleared {cleared} cached files")


# Example usage and testing
if __name__ == "__main__":
    from sp500_scraper import SP500Scraper
    
    # Initialize
    fetcher = StockDataFetcher()
    scraper = SP500Scraper()
    
    # Test single stock
    print("Testing single stock fetch...")
    df = fetcher.fetch_stock_data("AAPL", period="1mo")
    print(f"AAPL data shape: {df.shape}")
    print(df.head())
    
    # Add returns and volatility
    df = fetcher.calculate_returns(df)
    df = fetcher.calculate_volatility(df)
    print(f"\nColumns after calculations: {df.columns.tolist()}")
    
    # Test multiple stocks
    print("\n\nTesting multiple stock fetch...")
    test_tickers = ["MSFT", "GOOGL", "AMZN", "TSLA", "META"]
    results = fetcher.fetch_multiple_stocks(test_tickers, period="1mo")
    
    for ticker, data in results.items():
        print(f"{ticker}: {len(data)} days of data")
    
    # Test latest prices
    print("\n\nTesting latest prices...")
    latest = fetcher.get_latest_prices(test_tickers)
    print(latest)
    
    # Test stock info
    print("\n\nTesting stock info...")
    info = fetcher.get_stock_info("AAPL")
    for key, value in info.items():
        print(f"{key}: {value}")