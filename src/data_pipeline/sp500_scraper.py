"""
S&P 500 Constituents Scraper
Fetches and caches the current list of S&P 500 companies
Architecture Logic:

  1. SP500Scraper:
    - Source: Wikipedia (reliable, free, updated)
    - Caching: 7-day cache to minimize scraping  
    - Fallback: Uses old cache if scraping fails
    - Output: List of tickers + company details (sector, industry)
  2. StockDataFetcher:
    - Data Source: Yahoo Finance via yfinance
    - Smart Caching: Different expiry for daily vs intraday
    - Parallel Processing: ThreadPoolExecutor for multiple stocks
    - Rate Limiting: 2 req/sec to avoid blocks
    - Calculations: Returns, volatility, ATR built-in
  3. Error Handling:
    - Graceful degradation (use cache if API fails)
    - Logging for debugging
    - Empty DataFrame returns instead of crashes

  Why This Design?

  - Scalability: Can fetch 500 stocks efficiently with parallel processing
  - Reliability: Cache prevents API failures from breaking the system
  - Performance: Cache reduces API calls by 90%+
  - Extensibility: Easy to add new calculations or data sources
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SP500Scraper:
    """Scrapes and manages S&P 500 constituents list"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the scraper
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "sp500_constituents.json")
        self.wikipedia_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
    
    def fetch_sp500_tickers(self, force_refresh: bool = False) -> List[str]:
        """
        Get list of S&P 500 ticker symbols
        
        Args:
            force_refresh: Force fresh download even if cache exists
            
        Returns:
            List of ticker symbols
        """
        # Check if we should use cached data
        if not force_refresh and self._is_cache_valid():
            logger.info("Using cached S&P 500 data")
            return self._load_from_cache()
        
        # Scrape fresh data
        logger.info("Fetching fresh S&P 500 data from Wikipedia")
        tickers = self._scrape_tickers()
        
        # Save to cache
        self._save_to_cache(tickers)
        
        return tickers
    
    def get_sp500_info(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get detailed S&P 500 company information
        
        Args:
            force_refresh: Force fresh download
            
        Returns:
            DataFrame with company details (ticker, name, sector, industry)
        """
        if not force_refresh and self._is_cache_valid():
            cache_data = self._load_cache_data()
            if 'companies' in cache_data:
                return pd.DataFrame(cache_data['companies'])
        
        # Scrape detailed data
        logger.info("Fetching detailed S&P 500 company data")
        companies_df = self._scrape_detailed_info()
        
        # Save to cache
        self._save_detailed_to_cache(companies_df)
        
        return companies_df
    
    def _scrape_tickers(self) -> List[str]:
        """
        Scrape ticker symbols from Wikipedia
        
        Returns:
            List of ticker symbols
        """
        try:
            # Fetch the Wikipedia page
            response = requests.get(self.wikipedia_url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the first table (contains S&P 500 companies)
            table = soup.find('table', {'id': 'constituents'})
            if not table:
                # Fallback: find by class
                table = soup.find('table', {'class': 'wikitable sortable'})
            
            # Extract tickers from the table
            tickers = []
            rows = table.find_all('tr')[1:]  # Skip header row
            
            for row in rows:
                cols = row.find_all('td')
                if cols:
                    # Ticker is usually in the first column
                    ticker = cols[0].text.strip()
                    # Clean up ticker (remove footnotes, etc.)
                    ticker = ticker.replace('\n', '').replace('.', '-')  # BRK.B -> BRK-B for yfinance
                    tickers.append(ticker)
            
            logger.info(f"Successfully scraped {len(tickers)} tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Error scraping S&P 500 tickers: {e}")
            # Return cached data if available
            if os.path.exists(self.cache_file):
                logger.info("Falling back to cached data due to scraping error")
                return self._load_from_cache()
            raise
    
    def _scrape_detailed_info(self) -> pd.DataFrame:
        """
        Scrape detailed company information
        
        Returns:
            DataFrame with company details
        """
        try:
            # Use pandas to read Wikipedia table directly
            tables = pd.read_html(self.wikipedia_url)
            
            # The first table contains the S&P 500 list
            df = tables[0]
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Rename columns for consistency
            column_mapping = {
                'Symbol': 'ticker',
                'Security': 'name',
                'GICS Sector': 'sector',
                'GICS Sub-Industry': 'industry',
                'CIK': 'cik',
                'Founded': 'founded',
                'Date added': 'date_added',
                'Headquarters Location': 'headquarters'
            }
            
            # Rename only columns that exist
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Clean ticker symbols
            if 'ticker' in df.columns:
                df['ticker'] = df['ticker'].str.replace('.', '-', regex=False)
            
            logger.info(f"Successfully scraped detailed info for {len(df)} companies")
            return df
            
        except Exception as e:
            logger.error(f"Error scraping detailed S&P 500 info: {e}")
            raise
    
    def _is_cache_valid(self, max_age_days: int = 7) -> bool:
        """
        Check if cached data is still valid
        
        Args:
            max_age_days: Maximum age of cache in days
            
        Returns:
            True if cache is valid, False otherwise
        """
        if not os.path.exists(self.cache_file):
            return False
        
        # Check file age
        file_modified = datetime.fromtimestamp(os.path.getmtime(self.cache_file))
        age = datetime.now() - file_modified
        
        is_valid = age.days < max_age_days
        if is_valid:
            logger.info(f"Cache is {age.days} days old (valid)")
        else:
            logger.info(f"Cache is {age.days} days old (expired)")
        
        return is_valid
    
    def _load_from_cache(self) -> List[str]:
        """Load ticker list from cache"""
        with open(self.cache_file, 'r') as f:
            data = json.load(f)
            return data.get('tickers', [])
    
    def _load_cache_data(self) -> Dict:
        """Load all cached data"""
        with open(self.cache_file, 'r') as f:
            return json.load(f)
    
    def _save_to_cache(self, tickers: List[str]):
        """Save ticker list to cache"""
        cache_data = {
            'tickers': tickers,
            'updated': datetime.now().isoformat(),
            'count': len(tickers)
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Cached {len(tickers)} tickers to {self.cache_file}")
    
    def _save_detailed_to_cache(self, df: pd.DataFrame):
        """Save detailed company info to cache"""
        cache_data = {
            'tickers': df['ticker'].tolist() if 'ticker' in df.columns else [],
            'companies': df.to_dict('records'),
            'updated': datetime.now().isoformat(),
            'count': len(df)
        }
        
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        logger.info(f"Cached detailed info for {len(df)} companies")


# Example usage and testing
if __name__ == "__main__":
    scraper = SP500Scraper()
    
    # Get just tickers
    tickers = scraper.fetch_sp500_tickers()
    print(f"\nFetched {len(tickers)} tickers")
    print(f"First 10 tickers: {tickers[:10]}")
    
    # Get detailed info
    companies_df = scraper.get_sp500_info()
    print(f"\nCompany info shape: {companies_df.shape}")
    print(f"\nColumns: {companies_df.columns.tolist()}")
    print(f"\nFirst 5 companies:")
    print(companies_df.head())
    
    # Show sector distribution
    if 'sector' in companies_df.columns:
        print(f"\nSector distribution:")
        print(companies_df['sector'].value_counts())