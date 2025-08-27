# Data Pipeline Module 📊

## Overview
The data pipeline module is the foundation of the AI Trading Bot system, responsible for collecting, caching, and preprocessing all market data that feeds into the AI models and analysis components. It acts as the single source of truth for all stock market data used throughout the system.

## Module Components - Detailed Implementation

### 1. **fetcher.py** - Market Data Collection Engine
- **Purpose**: Downloads and intelligently caches stock market OHLCV data from Yahoo Finance
- **Core Class**: `StockDataFetcher`

#### How It Works:
1. **Smart Caching System**:
   - Generates unique cache keys based on ticker, dates, period, and interval
   - Cache expiry: 24 hours for daily data, 1 hour for intraday data
   - Falls back to expired cache if API fails (resilience)
   - Uses pickle format for fast serialization/deserialization

2. **Parallel Data Fetching**:
   ```python
   # Fetches multiple stocks simultaneously using ThreadPoolExecutor
   # Default: 5 workers, can fetch 500 stocks in ~5 minutes
   fetch_multiple_stocks(tickers, max_workers=5)
   ```

3. **Rate Limiting Implementation**:
   - Enforces 2 requests/second to Yahoo Finance
   - Uses time-based throttling between requests
   - Prevents IP bans and ensures reliable access

4. **Key Methods**:
   - `fetch_stock_data()`: Single stock with cache check → API call → cache save
   - `fetch_multiple_stocks()`: Parallel fetching with progress tracking
   - `get_latest_prices()`: Real-time prices during market hours
   - `calculate_returns()`: Automatic computation of 1d, 5d, 20d returns
   - `calculate_volatility()`: 20-day rolling standard deviation

5. **Error Handling Strategy**:
   - Primary: Try to fetch from Yahoo Finance
   - Fallback 1: Use cached data even if expired
   - Fallback 2: Return empty DataFrame (never crashes)
   - Logs all errors for debugging

### 2. **preprocessor.py** - Feature Engineering Factory
- **Purpose**: Transforms raw OHLCV data into 200+ AI-ready features
- **Core Class**: `DataPreprocessor`

#### Feature Categories Created:

1. **Price Features (50+ features)**:
   - **Ratios**: high/low, close/open, price position in daily range
   - **Returns**: 1, 2, 3, 5, 10, 20, 30-day returns (both simple and log)
   - **Gaps**: Opening gaps and gap percentages
   - **Candlestick Patterns**: Doji, hammer, shooting star detection
   - **Body Metrics**: Body size, upper/lower shadows, body-to-shadow ratios

2. **Volume Features (30+ features)**:
   - **Volume Changes**: 1, 5, 10, 20-day percentage changes
   - **Volume Ratios**: Current vs average volume (surge detection)
   - **Moving Averages**: 5, 10, 20, 50-day volume MAs
   - **Price-Volume Correlation**: 20-day rolling correlation
   - **On-Balance Volume (OBV)**: Cumulative volume-price trend

3. **Momentum Features (40+ features)**:
   - **RSI**: 7, 14, 21-day periods
   - **MACD**: Signal line, histogram, crossovers
   - **Stochastic**: %K and %D lines
   - **Rate of Change**: 5, 10, 20-day ROC
   - **Money Flow Index**: 14-day MFI
   - **Momentum Oscillator**: Various timeframes

4. **Volatility Features (30+ features)**:
   - **ATR**: 7, 14, 21-day Average True Range
   - **Bollinger Bands**: Upper, lower, width, position (10, 20, 30-day)
   - **Standard Deviation**: 5, 10, 20, 50-day rolling
   - **Parkinson Volatility**: High-low based volatility
   - **Garman-Klass**: More accurate volatility using OHLC

5. **Pattern Features (30+ features)**:
   - **Trend Detection**: Linear regression slopes (10, 20, 30-day)
   - **Higher Highs/Lower Lows**: Consecutive counts and flags
   - **Support/Resistance**: Touch counts and distance metrics
   - **Breakout Detection**: 20 and 50-day breakout flags
   - **Channel Positions**: Price position in various channels

6. **Market Microstructure (20+ features)**:
   - **Spread Metrics**: High-low spread, typical price
   - **VWAP**: Volume-weighted average price
   - **Time-based**: Day of week, month effects
   - **Accumulation/Distribution**: A/D line and oscillator

#### Data Normalization Process:
1. **StandardScaler** (default): Zero mean, unit variance
2. **MinMaxScaler**: Scale to [0, 1] range
3. **RobustScaler**: Handles outliers using median/IQR

#### Missing Data Handling:
- **Interpolation** (default): Linear interpolation between known values
- **Forward Fill**: Use last known value
- **Mean Imputation**: Replace with column mean
- **Smart Fallback**: Always ensures no NaN values remain

#### Sequence Creation for Time Series:
```python
# Creates sliding windows of 60 days for LSTM/Transformer models
create_sequences(data, sequence_length=60, prediction_horizon=5)
# Output shape: (samples, 60, 200+) for 3D tensor input
```

### 3. **sp500_scraper.py** - Universe Management System
- **Purpose**: Maintains current S&P 500 constituents list with metadata
- **Core Class**: `SP500Scraper`

#### Implementation Details:

1. **Web Scraping Process**:
   - Target: Wikipedia's S&P 500 page (reliable, always updated)
   - Parsing: BeautifulSoup extracts table data
   - Fields: Ticker, Company Name, Sector, Sub-Industry, Date Added
   - Backup: Falls back to cached JSON if Wikipedia changes/fails

2. **Intelligent Caching**:
   ```python
   # Cache invalidation logic
   if cache_age > 7 days or file_not_exists:
       fetch_from_wikipedia()
   else:
       load_from_cache()
   ```

3. **Data Structure Returned**:
   ```json
   {
     "AAPL": {
       "name": "Apple Inc.",
       "sector": "Technology",
       "industry": "Consumer Electronics",
       "date_added": "1982-11-30"
     },
     ...
   }
   ```

4. **Sector Classification**:
   - 11 GICS sectors tracked
   - Enables sector-based filtering and analysis
   - Useful for diversification and sector rotation strategies

5. **Update Mechanism**:
   - Automatic refresh every 7 days
   - Manual refresh via `force_update=True`
   - Handles ticker changes (additions/removals)
   - Logs all changes for audit trail

## Data Flow

```
Internet Sources                 Data Pipeline                    Downstream Systems
----------------                 -------------                    ------------------

Yahoo Finance -----> fetcher.py -----> Cache Storage -----> preprocessing.py
                         |                  ^                        |
                         |                  |                        v
Wikipedia ---------> sp500_scraper.py ------+              Feature Engineering
                                                                     |
                                                                     v
                                                            AI Models & Analysis
```

## How It Connects to the Project

### Input Provider
- **For Screening Module**: Provides raw price/volume data for initial filtering
- **For AI Models**: Supplies preprocessed features for pattern recognition and prediction
- **For Technical Analysis**: Delivers clean OHLCV data for indicator calculations

### Data Quality Assurance
- Ensures consistent data format across all modules
- Handles edge cases (weekends, holidays, missing data)
- Provides normalized data so different stocks can be compared fairly

### Performance Optimization
- Caching reduces API calls by 90%+
- Parallel processing enables analysis of 500+ stocks in minutes
- Smart expiry ensures data freshness while minimizing redundant fetches

## Key Design Decisions

1. **Why Cache Data?**
   - Yahoo Finance has rate limits
   - Reduces analysis time from hours to seconds
   - Enables backtesting on historical data without re-fetching

2. **Why 200+ Features?**
   - Raw OHLCV only has 5 features - not enough for ML
   - Different models need different feature types
   - More features allow models to find complex patterns

3. **Why Separate Fetching from Processing?**
   - Modularity - can swap data sources easily
   - Testability - can test preprocessing without API calls
   - Scalability - can add new data sources independently

## Usage Example

```python
from src.data_pipeline import StockDataFetcher, DataPreprocessor, SP500Scraper

# Get S&P 500 stocks
scraper = SP500Scraper()
sp500_stocks = scraper.get_sp500_tickers()

# Fetch market data
fetcher = StockDataFetcher()
aapl_data = fetcher.fetch_stock_data('AAPL', period='6mo')

# Preprocess for AI models
preprocessor = DataPreprocessor()
features = preprocessor.create_features(aapl_data)
# Now features contains 200+ engineered features ready for ML
```

## Detailed Implementation Workflows

### Data Fetching Workflow (fetcher.py)
```python
# Step-by-step process when fetch_stock_data('AAPL', period='20d') is called:

1. Generate Cache Key:
   cache_key = "AAPL_1d_20d"  # Format: {ticker}_{interval}_{period}
   
2. Check Cache:
   if cache_exists and age < 24_hours:
       return cached_data  # Skip API call
   
3. Rate Limit Check:
   wait_time = max(0, 0.5 - (current_time - last_request))
   time.sleep(wait_time)  # Ensures 2 req/sec max
   
4. API Call:
   yf.Ticker('AAPL').history(period='20d')
   # Returns DataFrame with columns: Open, High, Low, Close, Volume
   
5. Data Enhancement:
   df['Returns_1d'] = df['Close'].pct_change()
   df['Volatility'] = df['Returns_1d'].rolling(20).std()
   df['ATR'] = calculate_average_true_range(df)
   
6. Cache Save:
   df.to_pickle('data/cache/prices/AAPL_1d_20d.pkl')
   
7. Return Enhanced DataFrame
```

### Feature Engineering Workflow (preprocessor.py)
```python
# Transformation pipeline for raw OHLCV → 200+ features:

1. Input Validation:
   - Check for required columns (Open, High, Low, Close, Volume)
   - Ensure minimum 60 rows for proper feature calculation
   
2. Missing Data Handling:
   - Detect NaN values
   - Apply interpolation/forward-fill based on config
   - Log any remaining issues
   
3. Feature Creation Pipeline:
   raw_data → price_features → volume_features → momentum_features 
           → volatility_features → pattern_features → technical_features
   
4. Feature Matrix Assembly:
   all_features = pd.concat([
       price_features,    # 50+ columns
       volume_features,   # 30+ columns  
       momentum_features, # 40+ columns
       volatility_features, # 30+ columns
       pattern_features,  # 30+ columns
       technical_features # 20+ columns
   ], axis=1)
   
5. Normalization:
   scaler.fit_transform(all_features)  # Zero mean, unit variance
   
6. Sequence Creation (for LSTM/Transformer):
   sequences = []
   for i in range(60, len(data)):
       sequences.append(data[i-60:i])
   # Shape: (n_samples, 60_timesteps, 200_features)
   
7. Tensor Conversion:
   torch.tensor(sequences, dtype=torch.float32)
```

### Multi-Stock Processing Pipeline
```python
# How the system processes 500 stocks efficiently:

1. Get Universe:
   sp500_tickers = SP500Scraper().get_sp500_tickers()
   # Returns: ['AAPL', 'MSFT', 'GOOGL', ...] (500+ tickers)
   
2. Parallel Fetching:
   with ThreadPoolExecutor(max_workers=10) as executor:
       futures = {executor.submit(fetch, ticker): ticker 
                  for ticker in sp500_tickers}
       
3. Progressive Results:
   for future in as_completed(futures):
       ticker = futures[future]
       data = future.result()
       # Process immediately, don't wait for all
       
4. Feature Engineering:
   for ticker, raw_data in stock_data.items():
       features[ticker] = preprocessor.create_features(raw_data)
       
5. Quality Control:
   - Remove stocks with < 60 days history
   - Flag stocks with > 5% missing data
   - Log any processing errors
```

## Internal Data Structures

### Cache File Structure
```
data/cache/
├── prices/
│   ├── AAPL_1d_20d.pkl      # Daily data, 20 days
│   ├── AAPL_1h_5d.pkl        # Hourly data, 5 days
│   └── AAPL_5m_1d.pkl        # 5-min data, 1 day
├── sp500_constituents.json   # S&P 500 list
└── features/
    └── AAPL_features_20240101.pkl  # Preprocessed features
```

### Feature DataFrame Structure
```python
# Output from preprocessor.create_all_features():
DataFrame with 200+ columns:
- Index: DatetimeIndex
- Columns:
  - Price: returns_1d, returns_5d, gap_percentage, ...
  - Volume: volume_ratio_5d, obv, volume_ma_20, ...
  - Momentum: rsi_14, macd_signal, stochastic_k, ...
  - Volatility: atr_14, bb_upper_20, std_20, ...
  - Patterns: higher_high_10, breakout_up_20, trend_slope_30, ...
```

## Performance Optimizations

### 1. Caching Strategy
- **L1 Cache**: In-memory dictionary for current session
- **L2 Cache**: Pickle files on disk
- **Cache Keys**: Include all parameters to prevent collisions
- **Smart Expiry**: Different TTL for different data types

### 2. Parallel Processing
- **ThreadPoolExecutor**: I/O bound operations (API calls)
- **ProcessPoolExecutor**: CPU bound operations (feature engineering)
- **Batch Processing**: Process in chunks of 50 stocks
- **Progressive Loading**: Don't wait for all data before starting

### 3. Memory Management
- **Lazy Loading**: Load data only when needed
- **Garbage Collection**: Explicitly clear large DataFrames
- **Data Types**: Use float32 instead of float64 when possible
- **Chunking**: Process large datasets in chunks

### 4. API Optimization
- **Bulk Downloads**: Use yf.download() for multiple tickers
- **Minimal Requests**: Fetch maximum period in single request
- **Error Recovery**: Continue processing even if some tickers fail
- **Request Pooling**: Combine similar requests

## Error Handling & Recovery

### Common Issues & Solutions
1. **Yahoo Finance Rate Limit**
   - Solution: Exponential backoff, use cache, reduce workers
   
2. **Missing Data**
   - Solution: Interpolation, forward-fill, or skip stock
   
3. **Network Timeout**
   - Solution: Retry with exponential backoff, use cached data
   
4. **Invalid Ticker**
   - Solution: Log error, continue with remaining stocks
   
5. **Memory Overflow**
   - Solution: Process in smaller batches, clear intermediate results

## Performance Metrics
- Can fetch and cache 500 stocks in ~5 minutes
- Preprocessing adds ~0.1 seconds per stock
- Cache hit rate typically > 95% during trading hours
- Memory usage: ~2GB for full S&P 500 processing
- Feature creation: 200+ features in < 100ms per stock

## Dependencies
- **yfinance**: Yahoo Finance API wrapper
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **beautifulsoup4**: Web scraping for S&P 500 list
- **concurrent.futures**: Parallel processing
- **scikit-learn**: Normalization and imputation
- **torch**: Tensor conversion for neural networks
- **pickle**: Fast serialization for caching