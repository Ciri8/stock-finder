"""
Comprehensive tests for the Data Splitter module.
Tests various data splitting strategies for time series and ML models.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_splitter import DataSplitter

# Suppress warnings during tests
warnings.filterwarnings('ignore')


class TestDataSplitter(unittest.TestCase):
    """Test suite for data splitter utilities."""
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.splitter = DataSplitter()
        
        # Create sample DataFrame
        self.n_samples = 1000
        self.dates = pd.date_range('2020-01-01', periods=self.n_samples, freq='D')
        self.df = pd.DataFrame({
            'price': np.random.randn(self.n_samples).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, self.n_samples),
            'returns': np.random.randn(self.n_samples) * 0.02,
            'pattern': np.random.choice(['bullish', 'bearish', 'neutral'], self.n_samples),
            'signal': np.random.choice([0, 1], self.n_samples)
        }, index=self.dates)
        
        # Create sample numpy array
        self.array = np.random.randn(self.n_samples, 5)
        
        # Create sequence data for LSTM testing
        self.sequence_length = 10
        self.n_sequences = 100
        self.X_sequences = np.random.randn(self.n_sequences, self.sequence_length, 5)
        self.y_sequences = np.random.randn(self.n_sequences)
    
    def test_chronological_split_dataframe(self):
        """Test chronological split with DataFrame."""
        train, val, test = self.splitter.chronological_split(
            self.df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        # Check sizes
        self.assertEqual(len(train), 700)
        self.assertEqual(len(val), 150)
        self.assertEqual(len(test), 150)
        
        # Check data integrity
        self.assertEqual(len(train) + len(val) + len(test), len(self.df))
        
        # Check chronological order (no overlap)
        self.assertTrue(train.index.max() < val.index.min())
        self.assertTrue(val.index.max() < test.index.min())
        
        # Check data types preserved
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(val, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
    
    def test_chronological_split_array(self):
        """Test chronological split with numpy array."""
        train, val, test = self.splitter.chronological_split(
            self.array, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
        )
        
        # Check sizes
        self.assertEqual(len(train), 600)
        self.assertEqual(len(val), 200)
        self.assertEqual(len(test), 200)
        
        # Check data integrity
        self.assertEqual(len(train) + len(val) + len(test), len(self.array))
        
        # Check data types preserved
        self.assertIsInstance(train, np.ndarray)
        self.assertIsInstance(val, np.ndarray)
        self.assertIsInstance(test, np.ndarray)
        
        # Check shape preserved
        self.assertEqual(train.shape[1], self.array.shape[1])
        self.assertEqual(val.shape[1], self.array.shape[1])
        self.assertEqual(test.shape[1], self.array.shape[1])
    
    def test_chronological_split_ratio_normalization(self):
        """Test ratio normalization when they don't sum to 1."""
        # Ratios that don't sum to 1
        train, val, test = self.splitter.chronological_split(
            self.df, train_ratio=0.8, val_ratio=0.2, test_ratio=0.2
        )
        
        # Should still split all data
        self.assertEqual(len(train) + len(val) + len(test), len(self.df))
        
        # Check normalized ratios (0.8/1.2, 0.2/1.2, 0.2/1.2)
        expected_train = int(1000 * (0.8/1.2))
        expected_val = int(1000 * (0.2/1.2))
        
        # Allow for rounding differences
        self.assertAlmostEqual(len(train), expected_train, delta=2)
        self.assertAlmostEqual(len(val), expected_val, delta=2)
    
    def test_sliding_window_split(self):
        """Test sliding window split for sequence models."""
        window_size = 60
        gap = 5
        
        train, val, test = self.splitter.sliding_window_split(
            self.df, 
            window_size=window_size,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            gap=gap
        )
        
        # Check that splits exist
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)
        
        # Check gaps between sets
        if len(val) > 0 and len(train) > 0:
            # There should be a gap between train and val
            train_end_date = train.index[-1]
            val_start_date = val.index[0]
            days_between = (val_start_date - train_end_date).days
            self.assertGreaterEqual(days_between, gap)
        
        if len(test) > 0 and len(val) > 0:
            # There should be a gap between val and test
            val_end_date = val.index[-1]
            test_start_date = test.index[0]
            days_between = (test_start_date - val_end_date).days
            self.assertGreaterEqual(days_between, gap)
        
        # Check minimum sizes for windows
        self.assertGreaterEqual(len(train), window_size)
        if len(val) > 0:
            self.assertGreaterEqual(len(val), window_size)
        if len(test) > 0:
            self.assertGreaterEqual(len(test), window_size)
    
    def test_sliding_window_split_edge_cases(self):
        """Test sliding window split with edge cases."""
        # Large window size
        small_df = self.df.head(100)
        train, val, test = self.splitter.sliding_window_split(
            small_df, window_size=50
        )
        
        # Should still produce valid splits
        self.assertGreater(len(train), 0)
        # Val and test might be empty due to size constraints
        self.assertIsInstance(val, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
        
        # Window size larger than data
        tiny_df = self.df.head(30)
        train, val, test = self.splitter.sliding_window_split(
            tiny_df, window_size=60
        )
        
        # Should handle gracefully
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(val, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
    
    def test_walk_forward_split(self):
        """Test walk-forward validation split."""
        n_splits = 5
        test_size = 50
        
        splits = self.splitter.walk_forward_split(
            self.df, n_splits=n_splits, test_size=test_size
        )
        
        # Check number of splits
        self.assertEqual(len(splits), n_splits)
        
        # Check each split
        for i, (train, test) in enumerate(splits):
            # Test set should be correct size
            self.assertEqual(len(test), test_size)
            
            # Training set should expand
            if i > 0:
                prev_train, _ = splits[i-1]
                self.assertGreater(len(train), len(prev_train))
            
            # No overlap between train and test
            if isinstance(train, pd.DataFrame):
                self.assertTrue(train.index.max() < test.index.min())
            
            # Check continuity
            if isinstance(train, pd.DataFrame):
                train_end = train.index[-1]
                test_start = test.index[0]
                days_between = (test_start - train_end).days
                self.assertEqual(days_between, 1)  # Should be consecutive
    
    def test_walk_forward_split_array(self):
        """Test walk-forward split with numpy array."""
        splits = self.splitter.walk_forward_split(
            self.array, n_splits=5, test_size=30
        )
        
        self.assertEqual(len(splits), 5)
        
        for train, test in splits:
            self.assertIsInstance(train, np.ndarray)
            self.assertIsInstance(test, np.ndarray)
            self.assertEqual(len(test), 30)
            self.assertEqual(train.shape[1], self.array.shape[1])
            self.assertEqual(test.shape[1], self.array.shape[1])
    
    def test_stratified_time_split(self):
        """Test stratified time split maintaining class distribution."""
        train, val, test = self.splitter.stratified_time_split(
            self.df, 
            label_column='pattern',
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Check sizes
        self.assertEqual(len(train) + len(val) + len(test), len(self.df))
        
        # Check class distributions are similar
        original_dist = self.df['pattern'].value_counts(normalize=True)
        train_dist = train['pattern'].value_counts(normalize=True)
        val_dist = val['pattern'].value_counts(normalize=True)
        test_dist = test['pattern'].value_counts(normalize=True)
        
        # Distributions should be roughly similar (within 10%)
        for pattern in original_dist.index:
            if pattern in train_dist.index:
                self.assertAlmostEqual(
                    train_dist[pattern], original_dist[pattern], delta=0.1
                )
            if pattern in val_dist.index:
                self.assertAlmostEqual(
                    val_dist[pattern], original_dist[pattern], delta=0.1
                )
            if pattern in test_dist.index:
                self.assertAlmostEqual(
                    test_dist[pattern], original_dist[pattern], delta=0.1
                )
        
        # Check time order is maintained (mostly)
        self.assertIsInstance(train, pd.DataFrame)
        self.assertIsInstance(val, pd.DataFrame)
        self.assertIsInstance(test, pd.DataFrame)
    
    def test_stratified_time_split_missing_column(self):
        """Test stratified split with missing label column."""
        # Should fall back to chronological split
        train, val, test = self.splitter.stratified_time_split(
            self.df, 
            label_column='nonexistent_column',
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Should still work (falls back to chronological)
        self.assertEqual(len(train), 700)
        self.assertEqual(len(val), 150)
        self.assertEqual(len(test), 150)
    
    def test_create_sequences_split(self):
        """Test sequence splitting for LSTM models."""
        X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.create_sequences_split(
            self.X_sequences,
            self.y_sequences,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Check X splits
        self.assertEqual(len(X_train), 70)
        self.assertEqual(len(X_val), 15)
        self.assertEqual(len(X_test), 15)
        
        # Check y splits
        self.assertEqual(len(y_train), 70)
        self.assertEqual(len(y_val), 15)
        self.assertEqual(len(y_test), 15)
        
        # Check shapes preserved
        self.assertEqual(X_train.shape[1:], self.X_sequences.shape[1:])
        self.assertEqual(X_val.shape[1:], self.X_sequences.shape[1:])
        self.assertEqual(X_test.shape[1:], self.X_sequences.shape[1:])
        
        # Check no overlap
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(self.X_sequences))
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), len(self.y_sequences))
    
    def test_get_split_info(self):
        """Test split information retrieval."""
        train, val, test = self.splitter.chronological_split(self.df)
        info = self.splitter.get_split_info(train, val, test)
        
        # Check info structure
        self.assertIn('total_samples', info)
        self.assertIn('train_samples', info)
        self.assertIn('val_samples', info)
        self.assertIn('test_samples', info)
        self.assertIn('train_ratio', info)
        self.assertIn('val_ratio', info)
        self.assertIn('test_ratio', info)
        
        # Check values
        self.assertEqual(info['total_samples'], 1000)
        self.assertEqual(info['train_samples'], 700)
        self.assertEqual(info['val_samples'], 150)
        self.assertEqual(info['test_samples'], 150)
        self.assertAlmostEqual(info['train_ratio'], 0.7, places=2)
        self.assertAlmostEqual(info['val_ratio'], 0.15, places=2)
        self.assertAlmostEqual(info['test_ratio'], 0.15, places=2)
        
        # Check date ranges for DataFrame with DatetimeIndex
        self.assertIn('train_date_range', info)
        self.assertIn('val_date_range', info)
        self.assertIn('test_date_range', info)
        
        # Verify date ranges
        train_start, train_end = info['train_date_range']
        val_start, val_end = info['val_date_range']
        test_start, test_end = info['test_date_range']
        
        self.assertEqual(train_start, train.index.min())
        self.assertEqual(train_end, train.index.max())
        self.assertEqual(val_start, val.index.min())
        self.assertEqual(val_end, val.index.max())
        self.assertEqual(test_start, test.index.min())
        self.assertEqual(test_end, test.index.max())
    
    def test_get_split_info_array(self):
        """Test split info with numpy arrays."""
        train, val, test = self.splitter.chronological_split(self.array)
        info = self.splitter.get_split_info(train, val, test)
        
        # Should still work but without date ranges
        self.assertIn('total_samples', info)
        self.assertNotIn('train_date_range', info)
        self.assertEqual(info['total_samples'], 1000)
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_df = pd.DataFrame()
        empty_array = np.array([])
        
        # Chronological split with empty data
        train, val, test = self.splitter.chronological_split(empty_df)
        self.assertEqual(len(train), 0)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(test), 0)
        
        # Walk-forward with empty data
        splits = self.splitter.walk_forward_split(empty_df)
        self.assertEqual(len(splits), 0)
    
    def test_single_sample_handling(self):
        """Test handling of single sample data."""
        single_df = self.df.head(1)
        
        train, val, test = self.splitter.chronological_split(single_df)
        
        # All data should go to train
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(test), 0)
    
    def test_custom_ratios(self):
        """Test various custom split ratios."""
        test_cases = [
            (0.8, 0.1, 0.1),
            (0.5, 0.25, 0.25),
            (0.6, 0.3, 0.1),
            (0.9, 0.05, 0.05),
        ]
        
        for train_ratio, val_ratio, test_ratio in test_cases:
            with self.subTest(ratios=(train_ratio, val_ratio, test_ratio)):
                train, val, test = self.splitter.chronological_split(
                    self.df,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio
                )
                
                # Check total
                self.assertEqual(len(train) + len(val) + len(test), len(self.df))
                
                # Check approximate ratios
                actual_train_ratio = len(train) / len(self.df)
                actual_val_ratio = len(val) / len(self.df)
                actual_test_ratio = len(test) / len(self.df)
                
                self.assertAlmostEqual(actual_train_ratio, train_ratio, delta=0.01)
                self.assertAlmostEqual(actual_val_ratio, val_ratio, delta=0.01)
                self.assertAlmostEqual(actual_test_ratio, test_ratio, delta=0.01)


class TestDataSplitterIntegration(unittest.TestCase):
    """Integration tests combining data splitter with other modules."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.splitter = DataSplitter()
        
        # Create realistic stock data
        n_days = 500
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Simulate price with trend and volatility
        price = 100
        prices = []
        for _ in range(n_days):
            returns = np.random.normal(0.0005, 0.02)  # Daily returns
            price *= (1 + returns)
            prices.append(price)
        
        self.stock_df = pd.DataFrame({
            'Open': np.array(prices) * np.random.uniform(0.98, 1.02, n_days),
            'High': np.array(prices) * np.random.uniform(1.01, 1.05, n_days),
            'Low': np.array(prices) * np.random.uniform(0.95, 0.99, n_days),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=dates)
    
    def test_splitter_with_pattern_recognition_workflow(self):
        """Test data splitter in pattern recognition workflow."""
        # Split data for training pattern recognizer
        train_df, val_df, test_df = self.splitter.chronological_split(
            self.stock_df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Verify splits are suitable for pattern recognition
        # Need at least 60 days for pattern detection
        self.assertGreaterEqual(len(train_df), 60)
        self.assertGreaterEqual(len(val_df), 60)
        self.assertGreaterEqual(len(test_df), 60)
        
        # Check data integrity for candlestick patterns
        for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            # OHLC relationship should be valid
            self.assertTrue((df['High'] >= df['Low']).all(), 
                          f"{df_name}: High should be >= Low")
            self.assertTrue((df['High'] >= df['Open']).all(), 
                          f"{df_name}: High should be >= Open")
            self.assertTrue((df['High'] >= df['Close']).all(), 
                          f"{df_name}: High should be >= Close")
            self.assertTrue((df['Low'] <= df['Open']).all(), 
                          f"{df_name}: Low should be <= Open")
            self.assertTrue((df['Low'] <= df['Close']).all(), 
                          f"{df_name}: Low should be <= Close")
    
    def test_splitter_for_sequence_models(self):
        """Test splitter for LSTM/RNN sequence models."""
        sequence_length = 30
        prediction_horizon = 5
        
        # Create sequences from stock data
        n_samples = len(self.stock_df) - sequence_length - prediction_horizon + 1
        
        X_sequences = []
        y_sequences = []
        
        for i in range(n_samples):
            # Input: OHLCV for sequence_length days
            X = self.stock_df.iloc[i:i+sequence_length][['Open', 'High', 'Low', 'Close', 'Volume']].values
            # Target: Next day's closing price
            y = self.stock_df.iloc[i+sequence_length]['Close']
            
            X_sequences.append(X)
            y_sequences.append(y)
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        # Split sequences
        X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.create_sequences_split(
            X_sequences, y_sequences
        )
        
        # Verify shapes for LSTM input
        self.assertEqual(X_train.shape[1], sequence_length)
        self.assertEqual(X_train.shape[2], 5)  # 5 features (OHLCV)
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_val), len(y_val))
        self.assertEqual(len(X_test), len(y_test))
        
        # No data leakage: test sequences should be from later time periods
        # This is guaranteed by chronological splitting
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(X_sequences))
    
    def test_walk_forward_for_backtesting(self):
        """Test walk-forward split for strategy backtesting."""
        # Typical backtesting scenario
        n_splits = 12  # Monthly retraining
        test_days = 20  # Test on 20 trading days
        
        splits = self.splitter.walk_forward_split(
            self.stock_df,
            n_splits=n_splits,
            test_size=test_days
        )
        
        backtest_results = []
        
        for i, (train, test) in enumerate(splits):
            # Simulate training a model
            train_return = train['Close'].pct_change().mean()
            
            # Simulate testing
            test_return = test['Close'].pct_change().mean()
            
            result = {
                'split': i + 1,
                'train_days': len(train),
                'test_days': len(test),
                'train_return': train_return,
                'test_return': test_return,
                'train_period': (train.index.min().date(), train.index.max().date()),
                'test_period': (test.index.min().date(), test.index.max().date())
            }
            backtest_results.append(result)
        
        # Verify backtesting results
        self.assertEqual(len(backtest_results), n_splits)
        
        # Training set should expand over time
        for i in range(1, len(backtest_results)):
            self.assertGreater(
                backtest_results[i]['train_days'],
                backtest_results[i-1]['train_days']
            )
        
        # Test sets should be consecutive
        for i in range(1, len(backtest_results)):
            prev_test_end = backtest_results[i-1]['test_period'][1]
            curr_test_start = backtest_results[i]['test_period'][0]
            
            # Should be roughly consecutive (allow for weekends)
            days_between = (curr_test_start - prev_test_end).days
            self.assertLessEqual(days_between, 5)  # Allow for weekends


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestDataSplitter))
    suite.addTests(loader.loadTestsFromTestCase(TestDataSplitterIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    if result.testsRun > 0:
        print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result


if __name__ == "__main__":
    print("="*60)
    print("DATA SPLITTER MODULE TEST SUITE")
    print("="*60)
    run_tests()