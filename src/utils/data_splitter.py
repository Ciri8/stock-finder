"""
Data splitting utilities for train/validation/test sets.
Provides methods for both time series and standard data splitting.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Utility class for splitting data into train/validation/test sets."""
    
    @staticmethod
    def chronological_split(data: Union[pd.DataFrame, np.ndarray],
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15) -> Tuple:
        """
        Split time series data chronologically (no shuffling).
        Essential for stock data to prevent look-ahead bias.
        
        Args:
            data: Data to split (DataFrame or array)
            train_ratio: Proportion for training (default 0.7)
            val_ratio: Proportion for validation (default 0.15)
            test_ratio: Proportion for testing (default 0.15)
            
        Returns:
            Tuple of (train_data, val_data, test_data)
            
        Example:
            If you have 1000 days of data with 70/15/15 split:
            - Train: Days 1-700 (70%)
            - Val: Days 701-850 (15%)
            - Test: Days 851-1000 (15%)
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            logger.warning(f"Ratios sum to {total_ratio}, normalizing to 1.0")
            train_ratio = train_ratio / total_ratio
            val_ratio = val_ratio / total_ratio
            test_ratio = test_ratio / total_ratio
        
        n_samples = len(data)
        
        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split the data
        if isinstance(data, pd.DataFrame):
            train_data = data.iloc[:train_end]
            val_data = data.iloc[train_end:val_end]
            test_data = data.iloc[val_end:]
        else:
            train_data = data[:train_end]
            val_data = data[train_end:val_end]
            test_data = data[val_end:]
        
        logger.info(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    @staticmethod
    def sliding_window_split(data: Union[pd.DataFrame, np.ndarray],
                           window_size: int,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15,
                           gap: int = 0) -> Tuple:
        """
        Split data for sliding window time series models (like LSTM).
        Ensures no overlap between train/val/test windows.
        
        Args:
            data: Data to split
            window_size: Size of each window/sequence
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            gap: Number of samples to skip between sets (prevents leakage)
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n_samples = len(data)
        
        # Account for window size in the split
        effective_samples = n_samples - window_size + 1
        
        # Calculate split points
        train_end = int(effective_samples * train_ratio) + window_size - 1
        val_start = train_end + gap + 1
        val_end = val_start + int(effective_samples * val_ratio)
        test_start = val_end + gap + 1
        
        # Ensure we don't exceed data bounds
        val_end = min(val_end, n_samples)
        test_start = min(test_start, n_samples)
        
        # Split the data
        if isinstance(data, pd.DataFrame):
            train_data = data.iloc[:train_end]
            val_data = data.iloc[val_start:val_end] if val_start < n_samples else pd.DataFrame()
            test_data = data.iloc[test_start:] if test_start < n_samples else pd.DataFrame()
        else:
            train_data = data[:train_end]
            val_data = data[val_start:val_end] if val_start < n_samples else np.array([])
            test_data = data[test_start:] if test_start < n_samples else np.array([])
        
        logger.info(f"Sliding window split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        logger.info(f"Gap of {gap} samples between sets to prevent data leakage")
        
        return train_data, val_data, test_data
    
    @staticmethod
    def walk_forward_split(data: Union[pd.DataFrame, np.ndarray],
                          n_splits: int = 5,
                          test_size: int = 30) -> list:
        """
        Walk-forward validation for time series (expanding window).
        Used for robust backtesting of trading strategies.
        
        Args:
            data: Data to split
            n_splits: Number of splits to create
            test_size: Size of each test set
            
        Returns:
            List of (train_data, test_data) tuples
            
        Example:
            With 365 days of data, n_splits=5, test_size=30:
            Split 1: Train[0:185], Test[185:215]
            Split 2: Train[0:215], Test[215:245]
            Split 3: Train[0:245], Test[245:275]
            Split 4: Train[0:275], Test[275:305]
            Split 5: Train[0:305], Test[305:335]
        """
        n_samples = len(data)
        splits = []
        
        # Calculate the step size
        step_size = (n_samples - test_size) // n_splits
        
        for i in range(n_splits):
            # Expanding training window
            train_end = step_size * (i + 1)
            test_start = train_end
            test_end = min(test_start + test_size, n_samples)
            
            if test_end > n_samples:
                break
            
            if isinstance(data, pd.DataFrame):
                train_data = data.iloc[:train_end]
                test_data = data.iloc[test_start:test_end]
            else:
                train_data = data[:train_end]
                test_data = data[test_start:test_end]
            
            splits.append((train_data, test_data))
            
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    @staticmethod
    def stratified_time_split(data: pd.DataFrame,
                            label_column: str,
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15) -> Tuple:
        """
        Time-aware stratified split that maintains class distribution.
        Useful when you have imbalanced classes (e.g., rare patterns).
        
        Args:
            data: DataFrame with features and labels
            label_column: Name of the column containing labels
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if label_column not in data.columns:
            logger.warning(f"Label column '{label_column}' not found, using chronological split")
            return DataSplitter.chronological_split(data, train_ratio, val_ratio, test_ratio)
        
        # Group by label to maintain distribution
        unique_labels = data[label_column].unique()
        train_list, val_list, test_list = [], [], []
        
        for label in unique_labels:
            label_data = data[data[label_column] == label]
            
            # Split this label's data chronologically
            train_part, val_part, test_part = DataSplitter.chronological_split(
                label_data, train_ratio, val_ratio, test_ratio
            )
            
            train_list.append(train_part)
            val_list.append(val_part)
            test_list.append(test_part)
        
        # Combine and sort by index to maintain time order
        train_data = pd.concat(train_list).sort_index()
        val_data = pd.concat(val_list).sort_index()
        test_data = pd.concat(test_list).sort_index()
        
        # Log class distribution
        for name, dataset in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
            if len(dataset) > 0:
                dist = dataset[label_column].value_counts(normalize=True)
                logger.info(f"{name} set class distribution: {dist.to_dict()}")
        
        return train_data, val_data, test_data
    
    @staticmethod
    def create_sequences_split(X: np.ndarray, 
                             y: np.ndarray,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15) -> Tuple:
        """
        Split sequence data (already windowed) for LSTM/RNN models.
        
        Args:
            X: Input sequences (shape: [samples, sequence_length, features])
            y: Target values (shape: [samples] or [samples, outputs])
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)
        
        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split X
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        # Split y
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        logger.info(f"Sequence split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def get_split_info(train_data, val_data, test_data) -> dict:
        """
        Get information about the data splits.
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            
        Returns:
            Dictionary with split information
        """
        total_samples = len(train_data) + len(val_data) + len(test_data)
        
        info = {
            'total_samples': total_samples,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'train_ratio': len(train_data) / total_samples if total_samples > 0 else 0,
            'val_ratio': len(val_data) / total_samples if total_samples > 0 else 0,
            'test_ratio': len(test_data) / total_samples if total_samples > 0 else 0
        }
        
        # Add date range info if DataFrame with datetime index
        if isinstance(train_data, pd.DataFrame) and isinstance(train_data.index, pd.DatetimeIndex):
            info['train_date_range'] = (train_data.index.min(), train_data.index.max())
            if len(val_data) > 0:
                info['val_date_range'] = (val_data.index.min(), val_data.index.max())
            if len(test_data) > 0:
                info['test_date_range'] = (test_data.index.min(), test_data.index.max())
        
        return info


# Example usage
if __name__ == "__main__":
    # Create sample time series data
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    data = pd.DataFrame({
        'price': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 1000),
        'pattern': np.random.choice(['bullish', 'bearish', 'neutral'], 1000)
    }, index=dates)
    
    splitter = DataSplitter()
    
    print("="*60)
    print("1. Chronological Split (for general time series)")
    print("="*60)
    train, val, test = splitter.chronological_split(data)
    info = splitter.get_split_info(train, val, test)
    print(f"Train: {info['train_samples']} samples ({info['train_ratio']:.1%})")
    print(f"Val: {info['val_samples']} samples ({info['val_ratio']:.1%})")
    print(f"Test: {info['test_samples']} samples ({info['test_ratio']:.1%})")
    if 'train_date_range' in info:
        print(f"Train dates: {info['train_date_range'][0].date()} to {info['train_date_range'][1].date()}")
        print(f"Val dates: {info['val_date_range'][0].date()} to {info['val_date_range'][1].date()}")
        print(f"Test dates: {info['test_date_range'][0].date()} to {info['test_date_range'][1].date()}")
    
    print("\n" + "="*60)
    print("2. Sliding Window Split (for LSTM with sequences)")
    print("="*60)
    train, val, test = splitter.sliding_window_split(data, window_size=60, gap=5)
    print(f"With 60-day windows and 5-day gaps between sets:")
    print(f"Train: {len(train)} samples")
    print(f"Val: {len(val)} samples")
    print(f"Test: {len(test)} samples")
    
    print("\n" + "="*60)
    print("3. Walk-Forward Split (for backtesting)")
    print("="*60)
    splits = splitter.walk_forward_split(data, n_splits=5, test_size=50)
    for i, (train, test) in enumerate(splits, 1):
        print(f"Split {i}: Train={len(train)} samples, Test={len(test)} samples")
    
    print("\n" + "="*60)
    print("4. Stratified Time Split (for pattern detection)")
    print("="*60)
    train, val, test = splitter.stratified_time_split(data, label_column='pattern')
    print(f"Maintains pattern distribution across splits")
    print(f"Train: {len(train)} samples")
    print(f"Val: {len(val)} samples")
    print(f"Test: {len(test)} samples")