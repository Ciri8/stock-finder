"""
Test file:
test_preprocessor.py

Data preprocessing module for AI models.
Handles feature engineering, normalization, and data preparation for ML/DL models.


What is the Preprocessor?

  Think of it as a data chef that takes raw stock market data and prepares it for AI models to understand. Just like you can't feed raw
  ingredients to someone - you need to cook and season them first - AI models can't understand raw stock prices directly.

  What Raw Data Looks Like

  The stock market gives us basic data like:
  - Open: Price when market opened
  - High: Highest price that day
  - Low: Lowest price that day
  - Close: Price when market closed
  - Volume: How many shares were traded

  That's only 5 pieces of information per day. Not enough for smart AI predictions!

  What the Preprocessor Does

  1. Creates TONS of New Information (200+ features)

  From those 5 basic values, it calculates things like:

  - "Is the stock going up or down?" - Calculates momentum over different time periods
  - "Is it expensive or cheap right now?" - Compares current price to recent averages
  - "Are people excited about this stock?" - Looks at trading volume patterns
  - "Is this stock bouncing around a lot?" - Measures volatility
  - "Are we seeing any patterns?" - Detects things like "higher highs" (good sign) or "lower lows" (bad sign)

  2. Normalizes the Data

  Different stocks have different price ranges:
  - Apple might trade at $180
  - Berkshire Hathaway trades at $500,000+

  The preprocessor scales everything to similar ranges (like 0-1 or -1 to 1) so the AI doesn't get confused by the huge differences.

  3. Fills in Missing Data

  Sometimes data has gaps (holidays, halted trading, etc.). The preprocessor intelligently fills these gaps so the AI doesn't break.

  4. Creates Time Windows

  Instead of looking at one day at a time, it creates "sequences" - like looking at 60 days of history to predict what happens next. Like
  watching a movie instead of looking at a single photo.

  5. Converts to AI Format

  Transforms everything into "tensors" - the special format that AI models understand (think of it like translating English to AI language).

  Real World Example

  Raw Data: "Apple closed at $180 today with 50M volume"

  After Preprocessing:
  - "Apple is 5% above its 20-day average"
  - "Volume is 150% higher than normal"
  - "Stock has made higher highs for 3 days straight"
  - "RSI shows it's not overbought yet"
  - "Price is near the upper Bollinger Band"
  - "Momentum is positive and accelerating"
  - ... (200+ more insights)

  Why This Matters

  Without the preprocessor, the AI would be like a chef trying to make a meal with just salt and pepper. With it, the AI has a full spice rack       
  and all the ingredients it needs to make smart predictions about which stocks might go up or down.

  The preprocessor essentially turns simple price data into a rich story that AI can understand and learn from!
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedData:
    """Container for preprocessed data."""
    ticker: str
    features: pd.DataFrame
    feature_names: List[str]
    tensor_data: Optional[torch.Tensor] = None
    scaler: Optional[object] = None
    metadata: Optional[Dict] = None


class DataPreprocessor:
    """
    Comprehensive data preprocessing for AI trading models.
    Creates 200+ engineered features from OHLCV data.
    """
    
    def __init__(self, 
                 normalize_method: str = 'standard',
                 handle_missing: str = 'interpolate',
                 sequence_length: int = 60,
                 feature_engineering: bool = True):
        """
        Initialize the data preprocessor.
        
        Args:
            normalize_method: 'standard', 'minmax', or 'robust'
            handle_missing: 'interpolate', 'forward_fill', 'drop', or 'mean'
            sequence_length: Length of sequences for time series models
            feature_engineering: Whether to create engineered features
        """
        self.normalize_method = normalize_method
        self.handle_missing = handle_missing
        self.sequence_length = sequence_length
        self.feature_engineering = feature_engineering
        
        # Initialize scalers
        self.scaler = self._get_scaler(normalize_method)
        self.volume_scaler = RobustScaler()  # Robust for volume due to outliers
        
    def _get_scaler(self, method: str):
        """Get the appropriate scaler based on method."""
        if method == 'standard':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        elif method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data based on specified method.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if self.handle_missing == 'interpolate':
            # Interpolate missing values
            df = df.interpolate(method='linear', limit_direction='both')
        elif self.handle_missing == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif self.handle_missing == 'drop':
            df = df.dropna()
        elif self.handle_missing == 'mean':
            # Fill with mean for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
        
        # Final check - if still NaN, forward fill
        if df.isnull().any().any():
            df = df.fillna(method='ffill').fillna(method='bfill')
            
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=df.index)
        
        # Basic price ratios
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Price position within the day
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Gaps
        features['gap'] = df['open'] - df['close'].shift(1)
        features['gap_percentage'] = features['gap'] / df['close'].shift(1)
        
        # Price changes
        for period in [1, 2, 3, 5, 10, 20, 30]:
            features[f'return_{period}d'] = df['close'].pct_change(period)
            features[f'log_return_{period}d'] = np.log(df['close'] / df['close'].shift(period))
        
        # High/Low changes
        for period in [5, 10, 20]:
            features[f'high_change_{period}d'] = df['high'].pct_change(period)
            features[f'low_change_{period}d'] = df['low'].pct_change(period)
        
        # Candlestick patterns
        features['body_size'] = abs(df['close'] - df['open'])
        features['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
        features['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
        features['body_to_shadow_ratio'] = features['body_size'] / (features['upper_shadow'] + features['lower_shadow'] + 1e-10)
        
        # Doji detection (small body)
        features['is_doji'] = (features['body_size'] / (df['high'] - df['low'] + 1e-10) < 0.1).astype(int)
        
        # Hammer pattern (long lower shadow, small body at top)
        features['is_hammer'] = ((features['lower_shadow'] > 2 * features['body_size']) & 
                                 (features['upper_shadow'] < features['body_size'])).astype(int)
        
        return features
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features.
        
        Args:
            df: DataFrame with volume data
            
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=df.index)
        
        # Volume changes
        for period in [1, 5, 10, 20]:
            features[f'volume_change_{period}d'] = df['volume'].pct_change(period)
            features[f'volume_ratio_{period}d'] = df['volume'] / df['volume'].rolling(period).mean()
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            features[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            features[f'volume_std_{period}'] = df['volume'].rolling(period).std()
        
        # Price-Volume interaction
        features['price_volume_trend'] = (df['close'].pct_change() * df['volume']).rolling(10).sum()
        features['volume_price_correlation'] = df['close'].rolling(20).corr(df['volume'])
        
        # Accumulation/Distribution
        features['money_flow'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10) * df['volume']
        features['money_flow_ma'] = features['money_flow'].rolling(14).mean()
        
        # Volume spikes
        volume_mean = df['volume'].rolling(20).mean()
        volume_std = df['volume'].rolling(20).std()
        features['volume_spike'] = (df['volume'] - volume_mean) / (volume_std + 1e-10)
        
        return features
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create momentum-based features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=df.index)
        
        # Rate of Change (ROC)
        for period in [5, 10, 20, 30]:
            features[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
        
        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # Acceleration (momentum of momentum)
        features['price_acceleration'] = features['momentum_5'].diff()
        
        # Relative strength
        gains = df['close'].diff()
        gains[gains < 0] = 0
        losses = -df['close'].diff()
        losses[losses < 0] = 0
        
        for period in [7, 14, 21]:
            avg_gain = gains.rolling(period).mean()
            avg_loss = losses.rolling(period).mean()
            features[f'rs_{period}'] = avg_gain / (avg_loss + 1e-10)
        
        # Price efficiency
        for period in [10, 20]:
            net_change = df['close'] - df['close'].shift(period)
            total_movement = df['close'].diff().abs().rolling(period).sum()
            features[f'efficiency_{period}'] = net_change / (total_movement + 1e-10)
        
        return features
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create statistical features using rolling windows.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with statistical features
        """
        features = pd.DataFrame(index=df.index)
        
        # Rolling statistics for different periods
        for period in [5, 10, 20, 30, 60]:
            # Price statistics
            features[f'mean_{period}'] = df['close'].rolling(period).mean()
            features[f'std_{period}'] = df['close'].rolling(period).std()
            features[f'skew_{period}'] = df['close'].rolling(period).skew()
            features[f'kurt_{period}'] = df['close'].rolling(period).kurt()
            
            # Min/Max
            features[f'min_{period}'] = df['close'].rolling(period).min()
            features[f'max_{period}'] = df['close'].rolling(period).max()
            features[f'range_{period}'] = features[f'max_{period}'] - features[f'min_{period}']
            
            # Position in range
            features[f'position_in_range_{period}'] = (df['close'] - features[f'min_{period}']) / (features[f'range_{period}'] + 1e-10)
            
            # Quantiles
            features[f'quantile_25_{period}'] = df['close'].rolling(period).quantile(0.25)
            features[f'quantile_75_{period}'] = df['close'].rolling(period).quantile(0.75)
            features[f'iqr_{period}'] = features[f'quantile_75_{period}'] - features[f'quantile_25_{period}']
        
        # Z-score (standardized price)
        for period in [10, 20, 30]:
            mean = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'zscore_{period}'] = (df['close'] - mean) / (std + 1e-10)
        
        return features
    
    def create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create pattern-based features (higher highs/lows, etc).
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with pattern features
        """
        features = pd.DataFrame(index=df.index)
        
        # Higher highs and lower lows
        for period in [5, 10, 20]:
            # Rolling max/min
            rolling_high = df['high'].rolling(period).max()
            rolling_low = df['low'].rolling(period).min()
            
            # Higher highs
            features[f'higher_high_{period}'] = (df['high'] > rolling_high.shift(1)).astype(int)
            features[f'consecutive_higher_highs_{period}'] = features[f'higher_high_{period}'].groupby(
                (features[f'higher_high_{period}'] != features[f'higher_high_{period}'].shift()).cumsum()
            ).cumsum()
            
            # Lower lows
            features[f'lower_low_{period}'] = (df['low'] < rolling_low.shift(1)).astype(int)
            features[f'consecutive_lower_lows_{period}'] = features[f'lower_low_{period}'].groupby(
                (features[f'lower_low_{period}'] != features[f'lower_low_{period}'].shift()).cumsum()
            ).cumsum()
        
        # Trend detection
        for period in [10, 20, 30]:
            # Simple trend based on linear regression slope
            x = np.arange(period)
            slopes = []
            for i in range(len(df)):
                if i < period - 1:
                    slopes.append(np.nan)
                else:
                    y = df['close'].iloc[i-period+1:i+1].values
                    if len(y) == period:
                        slope = np.polyfit(x, y, 1)[0]
                        slopes.append(slope)
                    else:
                        slopes.append(np.nan)
            features[f'trend_slope_{period}'] = slopes
            features[f'trend_strength_{period}'] = np.abs(features[f'trend_slope_{period}']) / (df['close'].rolling(period).std() + 1e-10)
        
        # Support/Resistance levels
        for period in [20, 50]:
            # Count touches of recent highs/lows
            recent_high = df['high'].rolling(period).max()
            recent_low = df['low'].rolling(period).min()
            
            # How many times price touched these levels
            features[f'resistance_touches_{period}'] = (abs(df['high'] - recent_high) / recent_high < 0.01).rolling(period).sum()
            features[f'support_touches_{period}'] = (abs(df['low'] - recent_low) / recent_low < 0.01).rolling(period).sum()
        
        # Breakout detection
        for period in [20, 50]:
            features[f'breakout_up_{period}'] = (df['close'] > df['high'].rolling(period).max().shift(1)).astype(int)
            features[f'breakout_down_{period}'] = (df['close'] < df['low'].rolling(period).min().shift(1)).astype(int)
        
        return features
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicator features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features
        """
        features = pd.DataFrame(index=df.index)
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = df['close'].rolling(period).mean()
            features[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # Price relative to MA
            features[f'close_to_sma_{period}'] = df['close'] / features[f'sma_{period}']
            features[f'close_to_ema_{period}'] = df['close'] / features[f'ema_{period}']
        
        # Moving average crossovers
        features['ma_cross_5_20'] = (features['sma_5'] > features['sma_20']).astype(int)
        features['ma_cross_20_50'] = (features['sma_20'] > features['sma_50']).astype(int)
        features['ma_cross_50_200'] = (features['sma_50'] > features['sma_200']).astype(int)
        
        # Bollinger Bands
        for period in [10, 20, 30]:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = ma + (2 * std)
            features[f'bb_lower_{period}'] = ma - (2 * std)
            features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
            features[f'bb_position_{period}'] = (df['close'] - features[f'bb_lower_{period}']) / (features[f'bb_width_{period}'] + 1e-10)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        for period in [7, 14, 21]:
            features[f'atr_{period}'] = true_range.rolling(period).mean()
            features[f'atr_ratio_{period}'] = features[f'atr_{period}'] / df['close']
        
        return features
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical time-based features.
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with cyclical features
        """
        features = pd.DataFrame(index=df.index)
        
        if isinstance(df.index, pd.DatetimeIndex):
            # Day of week (cyclical encoding)
            day_of_week = df.index.dayofweek
            features['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            features['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Day of month (cyclical encoding)
            day_of_month = df.index.day
            features['month_day_sin'] = np.sin(2 * np.pi * day_of_month / 30)
            features['month_day_cos'] = np.cos(2 * np.pi * day_of_month / 30)
            
            # Month of year (cyclical encoding)
            month = df.index.month
            features['month_sin'] = np.sin(2 * np.pi * month / 12)
            features['month_cos'] = np.cos(2 * np.pi * month / 12)
            
            # Quarter
            features['quarter'] = df.index.quarter
            
            # Is month end/start
            features['is_month_start'] = df.index.is_month_start.astype(int)
            features['is_month_end'] = df.index.is_month_end.astype(int)
            features['is_quarter_start'] = df.index.is_quarter_start.astype(int)
            features['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return features
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features
        """
        # Ensure columns are lowercase
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Create all feature sets
        price_features = self.create_price_features(df)
        volume_features = self.create_volume_features(df)
        momentum_features = self.create_momentum_features(df)
        statistical_features = self.create_statistical_features(df)
        pattern_features = self.create_pattern_features(df)
        technical_features = self.create_technical_features(df)
        cyclical_features = self.create_cyclical_features(df)
        
        # Combine all features
        all_features = pd.concat([
            df[['open', 'high', 'low', 'close', 'volume']],
            price_features,
            volume_features,
            momentum_features,
            statistical_features,
            pattern_features,
            technical_features,
            cyclical_features
        ], axis=1)
        
        # Handle missing values
        all_features = self.handle_missing_data(all_features)
        
        return all_features
    
    def normalize_features(self, features: pd.DataFrame, 
                          fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features using specified method.
        
        Args:
            features: DataFrame with features
            fit: Whether to fit the scaler (True for training)
            
        Returns:
            Tuple of (normalized_features, scaler_dict)
        """
        normalized = features.copy()
        scalers = {}
        
        # Separate price and volume features
        price_cols = [col for col in features.columns if 'volume' not in col.lower()]
        volume_cols = [col for col in features.columns if 'volume' in col.lower()]
        
        # Normalize price features
        if price_cols:
            if fit:
                price_scaled = self.scaler.fit_transform(features[price_cols])
                scalers['price_scaler'] = self.scaler
            else:
                price_scaled = self.scaler.transform(features[price_cols])
            normalized[price_cols] = price_scaled
        
        # Normalize volume features (using robust scaler)
        if volume_cols:
            if fit:
                volume_scaled = self.volume_scaler.fit_transform(features[volume_cols])
                scalers['volume_scaler'] = self.volume_scaler
            else:
                volume_scaled = self.volume_scaler.transform(features[volume_cols])
            normalized[volume_cols] = volume_scaled
        
        return normalized, scalers
    
    def create_sequences(self, features: pd.DataFrame, 
                        target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series models.
        
        Args:
            features: DataFrame with features
            target_col: Column to use as target
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        data = features.values
        target_idx = features.columns.get_loc(target_col) if target_col in features.columns else -1
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            if target_idx >= 0:
                y.append(data[i, target_idx])
            else:
                y.append(data[i, 3])  # Default to close price
        
        return np.array(X), np.array(y)
    
    def to_tensor(self, data: Union[pd.DataFrame, np.ndarray]) -> torch.Tensor:
        """
        Convert data to PyTorch tensor.
        
        Args:
            data: DataFrame or numpy array
            
        Returns:
            PyTorch tensor
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        return torch.FloatTensor(data)
    
    def preprocess_stock(self, df: pd.DataFrame, 
                        ticker: str = '',
                        fit_scaler: bool = True) -> PreprocessedData:
        """
        Complete preprocessing pipeline for a single stock.
        
        Args:
            df: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            fit_scaler: Whether to fit the scaler
            
        Returns:
            PreprocessedData object
        """
        # Create all features
        if self.feature_engineering:
            features = self.create_all_features(df)
        else:
            features = df[['open', 'high', 'low', 'close', 'volume']].copy()
            features = self.handle_missing_data(features)
        
        # Normalize features
        normalized_features, scalers = self.normalize_features(features, fit=fit_scaler)
        
        # Create sequences for time series
        X_sequences, y_sequences = self.create_sequences(normalized_features)
        
        # Convert to tensor
        tensor_data = self.to_tensor(X_sequences) if len(X_sequences) > 0 else None
        
        # Create metadata
        metadata = {
            'num_features': features.shape[1],
            'num_samples': len(features),
            'sequence_length': self.sequence_length,
            'feature_stats': features.describe().to_dict(),
            'missing_values': features.isnull().sum().to_dict(),
            'sequences_shape': X_sequences.shape if len(X_sequences) > 0 else None
        }
        
        return PreprocessedData(
            ticker=ticker,
            features=normalized_features,
            feature_names=list(features.columns),
            tensor_data=tensor_data,
            scaler=scalers.get('price_scaler'),
            metadata=metadata
        )
    
    def preprocess_multiple_stocks(self, 
                                 stock_data: Dict[str, pd.DataFrame],
                                 max_workers: int = 10) -> Dict[str, PreprocessedData]:
        """
        Preprocess multiple stocks in parallel.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping ticker to PreprocessedData
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.preprocess_stock, df, ticker, True): ticker 
                for ticker, df in stock_data.items()
            }
            
            # Collect results
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    preprocessed = future.result()
                    results[ticker] = preprocessed
                    logger.info(f"Preprocessed {ticker}: {preprocessed.metadata['num_features']} features")
                except Exception as e:
                    logger.error(f"Error preprocessing {ticker}: {str(e)}")
                    results[ticker] = PreprocessedData(ticker=ticker, features=pd.DataFrame(), feature_names=[])
        
        return results
    
    def get_feature_importance(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic feature importance using correlation with returns.
        
        Args:
            features: DataFrame with features
            
        Returns:
            DataFrame with feature importance scores
        """
        if 'close' not in features.columns:
            return pd.DataFrame()
        
        # Calculate returns
        returns = features['close'].pct_change().shift(-1)  # Next period returns
        
        # Calculate correlation with returns
        correlations = features.corrwith(returns).abs()
        
        # Create importance dataframe
        importance = pd.DataFrame({
            'feature': correlations.index,
            'importance': correlations.values
        }).sort_values('importance', ascending=False)
        
        # Add feature category
        def categorize_feature(name):
            if 'volume' in name.lower():
                return 'volume'
            elif 'momentum' in name or 'roc' in name:
                return 'momentum'
            elif 'ma' in name or 'sma' in name or 'ema' in name:
                return 'moving_average'
            elif 'bb' in name:
                return 'bollinger'
            elif 'pattern' in name or 'higher' in name or 'lower' in name:
                return 'pattern'
            elif 'stat' in name or 'std' in name or 'mean' in name:
                return 'statistical'
            elif 'sin' in name or 'cos' in name:
                return 'cyclical'
            else:
                return 'price'
        
        importance['category'] = importance['feature'].apply(categorize_feature)
        
        return importance


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    import yfinance as yf
    
    # Fetch sample data
    ticker = "AAPL"
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        normalize_method='standard',
        handle_missing='interpolate',
        sequence_length=60,
        feature_engineering=True
    )
    
    # Preprocess the data
    preprocessed = preprocessor.preprocess_stock(df, ticker)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Preprocessed Data for {ticker}")
    print(f"{'='*60}")
    print(f"Number of features: {preprocessed.metadata['num_features']}")
    print(f"Number of samples: {preprocessed.metadata['num_samples']}")
    print(f"Sequence shape: {preprocessed.metadata['sequences_shape']}")
    print(f"\nFeature categories:")
    
    # Count features by category
    feature_categories = {}
    for feature in preprocessed.feature_names:
        if 'volume' in feature.lower():
            cat = 'Volume'
        elif 'momentum' in feature or 'roc' in feature:
            cat = 'Momentum'
        elif 'ma_' in feature or 'sma_' in feature or 'ema_' in feature:
            cat = 'Moving Average'
        elif 'bb_' in feature:
            cat = 'Bollinger Bands'
        elif 'higher' in feature or 'lower' in feature or 'pattern' in feature:
            cat = 'Pattern'
        elif 'std_' in feature or 'mean_' in feature or 'stat' in feature:
            cat = 'Statistical'
        elif 'sin' in feature or 'cos' in feature:
            cat = 'Cyclical'
        else:
            cat = 'Price/Other'
        
        feature_categories[cat] = feature_categories.get(cat, 0) + 1
    
    for cat, count in sorted(feature_categories.items()):
        print(f"  {cat}: {count} features")
    
    print(f"\nTotal features created: {sum(feature_categories.values())}")
    
    # Show sample of normalized data
    print(f"\nSample of normalized features (first 5 rows, first 10 columns):")
    print(preprocessed.features.iloc[:5, :10])
    
    # Get feature importance
    importance = preprocessor.get_feature_importance(preprocessor.create_all_features(df))
    print(f"\nTop 10 most important features:")
    print(importance.head(10))