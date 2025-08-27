"""
Integration tests combining Pattern Recognition and Data Splitter modules.
Tests the full ML pipeline workflow from data splitting to pattern detection.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import warnings
import tempfile
import shutil
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pattern_rec.pattern_recognition import ChartPatternRecognizer, ChartPatternDataset, PatternDetection
from src.utils.data_splitter import DataSplitter

warnings.filterwarnings('ignore')


class TestPatternRecognitionPipeline(unittest.TestCase):
    """Test full pipeline from data splitting to pattern detection."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {cls.device}")
        print(f"Temp directory: {cls.temp_dir}")
        
    @classmethod
    def tearDownClass(cls):
        """Clean up."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up for each test."""
        self.splitter = DataSplitter()
        self.recognizer = ChartPatternRecognizer(device=str(self.device))
        self.market_data = self._generate_market_data()
        
    def _generate_market_data(self, n_stocks=5, n_days=365):
        """Generate realistic market data for multiple stocks."""
        stocks = {}
        
        for i in range(n_stocks):
            ticker = f"STOCK_{chr(65+i)}"  # STOCK_A, STOCK_B, etc.
            
            # Generate correlated market data
            dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
            
            # Base trend (market factor)
            market_trend = np.cumsum(np.random.normal(0.001, 0.02, n_days))
            
            # Stock-specific movement
            stock_noise = np.cumsum(np.random.normal(0, 0.015, n_days))
            
            # Combine for realistic price movement
            prices = 100 * np.exp(market_trend + stock_noise)
            
            # Add patterns to some stocks
            if i == 0:  # Bull flag pattern
                prices[100:130] *= np.linspace(1.0, 1.15, 30)  # Sharp rise
                prices[130:160] *= np.linspace(1.15, 1.13, 30)  # Consolidation
                prices[160:180] *= np.linspace(1.13, 1.20, 20)  # Breakout
            elif i == 1:  # Cup and handle
                prices[50:100] *= np.linspace(1.0, 0.85, 50)  # Decline
                prices[100:150] *= np.linspace(0.85, 0.85, 50)  # Bottom
                prices[150:200] *= np.linspace(0.85, 1.0, 50)  # Recovery
                prices[200:220] *= np.linspace(1.0, 0.97, 20)  # Handle
                
            # Generate OHLCV
            data = []
            for j, (date, price) in enumerate(zip(dates, prices)):
                daily_vol = 0.02
                open_price = price * (1 + np.random.normal(0, daily_vol/2))
                close_price = price * (1 + np.random.normal(0, daily_vol/2))
                high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, daily_vol)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, daily_vol)))
                volume = int(5000000 * (1 + np.random.normal(0, 0.3)))
                
                data.append({
                    'Date': date,
                    'Open': open_price,
                    'High': high_price,
                    'Low': low_price,
                    'Close': close_price,
                    'Volume': volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            
            # Add technical indicators
            df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
            df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20, min_periods=1).std()
            
            stocks[ticker] = df
            
        return stocks
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline with data splitting."""
        print("\n" + "="*60)
        print("Testing Full Training Pipeline")
        print("="*60)
        
        # Step 1: Prepare combined dataset
        all_images = []
        all_labels = []
        
        for ticker, df in self.market_data.items():
            # Split each stock's data
            train_df, val_df, test_df = self.splitter.chronological_split(
                df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
            )
            
            print(f"\n{ticker} splits:")
            print(f"  Train: {len(train_df)} days")
            print(f"  Val: {len(val_df)} days")
            print(f"  Test: {len(test_df)} days")
            
            # Generate training images (sliding windows)
            window_size = 60
            if len(train_df) >= window_size:
                for i in range(0, len(train_df) - window_size, 10):  # Step by 10 days
                    window = train_df.iloc[i:i+window_size]
                    image = self.recognizer.ohlcv_to_image(window, window_size=window_size)
                    all_images.append(image)
                    # Assign synthetic labels for testing
                    label = hash(ticker) % 5  # Distribute among 5 pattern types
                    all_labels.append(label)
        
        # Convert to numpy arrays
        if all_images:
            images = np.array(all_images)
            labels = np.array(all_labels)
            
            print(f"\nDataset created:")
            print(f"  Total images: {len(images)}")
            print(f"  Image shape: {images[0].shape}")
            print(f"  Unique labels: {np.unique(labels)}")
            
            # Step 2: Split the image dataset
            X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.create_sequences_split(
                images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
            )
            
            print(f"\nImage dataset splits:")
            print(f"  Train: {len(X_train)} images")
            print(f"  Val: {len(X_val)} images")
            print(f"  Test: {len(X_test)} images")
            
            # Step 3: Create PyTorch datasets
            train_dataset = ChartPatternDataset(X_train, y_train)
            val_dataset = ChartPatternDataset(X_val, y_val)
            
            # Verify datasets
            self.assertEqual(len(train_dataset), len(X_train))
            self.assertEqual(len(val_dataset), len(X_val))
            
            # Test data loading
            sample_image, sample_label = train_dataset[0]
            self.assertEqual(sample_image.shape, (3, 224, 224))
    
    def test_walk_forward_pattern_detection(self):
        """Test walk-forward validation with pattern detection."""
        print("\n" + "="*60)
        print("Testing Walk-Forward Pattern Detection")
        print("="*60)
        
        # Use one stock for walk-forward testing
        stock_df = self.market_data['STOCK_A']
        
        # Create walk-forward splits
        splits = self.splitter.walk_forward_split(
            stock_df, n_splits=5, test_size=30
        )
        
        all_results = []
        
        for i, (train_df, test_df) in enumerate(splits):
            print(f"\nSplit {i+1}:")
            print(f"  Training period: {train_df.index[0].date()} to {train_df.index[-1].date()}")
            print(f"  Testing period: {test_df.index[0].date()} to {test_df.index[-1].date()}")
            
            # Detect patterns in test period
            if len(test_df) >= 60:
                patterns = self.recognizer.detect_patterns(test_df, confidence_threshold=0.1)
                
                result = {
                    'split': i + 1,
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    'patterns_detected': len(patterns),
                    'pattern_types': [p.pattern_type for p in patterns],
                    'avg_confidence': np.mean([p.confidence for p in patterns]) if patterns else 0
                }
                all_results.append(result)
                
                if patterns:
                    print(f"  Patterns detected: {len(patterns)}")
                    for p in patterns[:3]:  # Show top 3
                        print(f"    - {p.pattern_type}: {p.confidence:.1%}")
                else:
                    print("  No patterns detected")
        
        # Verify walk-forward results
        self.assertEqual(len(all_results), len(splits))
        
        # Analyze consistency across splits
        if all_results:
            avg_patterns = np.mean([r['patterns_detected'] for r in all_results])
            print(f"\nAverage patterns per split: {avg_patterns:.1f}")
    
    def test_multi_stock_batch_processing(self):
        """Test batch processing with proper data splitting."""
        print("\n" + "="*60)
        print("Testing Multi-Stock Batch Processing")
        print("="*60)
        
        # Split data for each stock
        train_data = {}
        test_data = {}
        
        for ticker, df in self.market_data.items():
            train_df, val_df, test_df = self.splitter.chronological_split(
                df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
            )
            train_data[ticker] = train_df
            test_data[ticker] = test_df
        
        # Batch detect patterns on test data
        test_results = self.recognizer.batch_detect(test_data, confidence_threshold=0.1)
        
        print("\nTest Set Pattern Detection Results:")
        total_patterns = 0
        for ticker, patterns in test_results.items():
            print(f"\n{ticker}:")
            print(f"  Test period: {test_data[ticker].index[0].date()} to {test_data[ticker].index[-1].date()}")
            print(f"  Patterns detected: {len(patterns)}")
            total_patterns += len(patterns)
            
            if patterns:
                for p in patterns[:2]:  # Show top 2
                    print(f"    - {p.pattern_type}: {p.confidence:.1%}")
                    if p.breakout_point:
                        print(f"      Breakout at: {test_data[ticker].index[p.breakout_point].date()}")
        
        print(f"\nTotal patterns across all stocks: {total_patterns}")
        
        # Verify results structure
        self.assertEqual(len(test_results), len(test_data))
        for ticker in test_data.keys():
            self.assertIn(ticker, test_results)
            self.assertIsInstance(test_results[ticker], list)
    
    def test_pattern_detection_with_stratified_split(self):
        """Test pattern detection with stratified splitting."""
        print("\n" + "="*60)
        print("Testing Stratified Split for Pattern Detection")
        print("="*60)
        
        # Add pattern labels to data
        stock_df = self.market_data['STOCK_A'].copy()
        
        # Simulate pattern labels (in real scenario, these would be from manual labeling)
        pattern_labels = []
        for i in range(len(stock_df)):
            if i % 100 < 20:
                pattern_labels.append('bullish')
            elif i % 100 < 40:
                pattern_labels.append('bearish')
            else:
                pattern_labels.append('neutral')
        
        stock_df['pattern_label'] = pattern_labels
        
        # Stratified split
        train_df, val_df, test_df = self.splitter.stratified_time_split(
            stock_df, label_column='pattern_label'
        )
        
        # Check label distribution
        print("\nLabel distributions:")
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            if len(df) > 0:
                dist = df['pattern_label'].value_counts(normalize=True)
                print(f"\n{name}:")
                for label, pct in dist.items():
                    print(f"  {label}: {pct:.1%}")
        
        # Detect patterns in each split
        for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            if len(df) >= 60:
                patterns = self.recognizer.detect_patterns(df, confidence_threshold=0.1)
                print(f"\n{name} patterns: {len(patterns)} detected")
    
    def test_sliding_window_for_lstm_preparation(self):
        """Test sliding window split for LSTM data preparation."""
        print("\n" + "="*60)
        print("Testing Sliding Window for LSTM Preparation")
        print("="*60)
        
        stock_df = self.market_data['STOCK_A']
        window_size = 60
        gap = 5
        
        # Split with sliding window
        train_df, val_df, test_df = self.splitter.sliding_window_split(
            stock_df, window_size=window_size, gap=gap
        )
        
        print(f"\nSliding window splits (window={window_size}, gap={gap}):")
        print(f"  Train: {len(train_df)} days")
        print(f"  Val: {len(val_df)} days")
        print(f"  Test: {len(test_df)} days")
        
        # Create sequences from each split
        def create_lstm_sequences(df, window_size, prediction_horizon=1):
            """Helper to create LSTM sequences."""
            sequences = []
            targets = []
            
            for i in range(len(df) - window_size - prediction_horizon + 1):
                # Extract window
                window = df.iloc[i:i+window_size]
                
                # Create image for pattern recognition
                image = self.recognizer.ohlcv_to_image(window)
                sequences.append(image)
                
                # Target is next day's return
                target = df.iloc[i+window_size]['Returns']
                targets.append(target)
            
            return np.array(sequences), np.array(targets)
        
        # Generate sequences for each split
        if len(train_df) > window_size:
            X_train, y_train = create_lstm_sequences(train_df, window_size)
            print(f"\nTrain sequences: {X_train.shape}")
            
            # Verify no overlap
            train_end_date = train_df.index[-1]
            if len(val_df) > 0:
                val_start_date = val_df.index[0]
                days_gap = (val_start_date - train_end_date).days
                print(f"Gap between train and val: {days_gap} days")
                self.assertGreaterEqual(days_gap, gap)
    
    def test_model_persistence_across_splits(self):
        """Test saving and loading models across different data splits."""
        print("\n" + "="*60)
        print("Testing Model Persistence Across Splits")
        print("="*60)
        
        stock_df = self.market_data['STOCK_A']
        
        # Initial split and training
        train_df, val_df, test_df = self.splitter.chronological_split(stock_df)
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'pattern_model.pth')
        self.recognizer.save_model(model_path)
        print(f"Model saved to: {model_path}")
        
        # Create new recognizer and load model
        new_recognizer = ChartPatternRecognizer(model_path=model_path)
        
        # Test on different splits
        splits = self.splitter.walk_forward_split(stock_df, n_splits=3, test_size=30)
        
        for i, (train_split, test_split) in enumerate(splits):
            if len(test_split) >= 60:
                # Use loaded model for detection
                patterns = new_recognizer.detect_patterns(test_split, confidence_threshold=0.1)
                print(f"\nSplit {i+1}: {len(patterns)} patterns detected with loaded model")
    
    def test_performance_metrics_collection(self):
        """Test collecting performance metrics across pipeline."""
        print("\n" + "="*60)
        print("Testing Performance Metrics Collection")
        print("="*60)
        
        metrics = {
            'data_splitting': {},
            'pattern_detection': {},
            'overall': {}
        }
        
        # Measure data splitting performance
        import time
        
        stock_df = self.market_data['STOCK_A']
        
        start_time = time.time()
        train_df, val_df, test_df = self.splitter.chronological_split(stock_df)
        split_time = time.time() - start_time
        
        metrics['data_splitting'] = {
            'time_seconds': split_time,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df)
        }
        
        # Measure pattern detection performance
        start_time = time.time()
        patterns = self.recognizer.detect_patterns(test_df, confidence_threshold=0.3)
        detect_time = time.time() - start_time
        
        metrics['pattern_detection'] = {
            'time_seconds': detect_time,
            'patterns_found': len(patterns),
            'confidence_scores': [p.confidence for p in patterns] if patterns else []
        }
        
        # Overall metrics
        metrics['overall'] = {
            'total_time': split_time + detect_time,
            'data_points_processed': len(stock_df),
            'efficiency': len(stock_df) / (split_time + detect_time) if (split_time + detect_time) > 0 else 0
        }
        
        # Save metrics
        metrics_path = os.path.join(self.temp_dir, 'pipeline_metrics.json')
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            for key in metrics:
                for subkey in metrics[key]:
                    if isinstance(metrics[key][subkey], np.ndarray):
                        metrics[key][subkey] = metrics[key][subkey].tolist()
            json.dump(metrics, f, indent=2)
        
        print("\nPipeline Metrics:")
        print(f"  Data splitting: {metrics['data_splitting']['time_seconds']:.4f}s")
        print(f"  Pattern detection: {metrics['pattern_detection']['time_seconds']:.4f}s")
        print(f"  Total time: {metrics['overall']['total_time']:.4f}s")
        print(f"  Efficiency: {metrics['overall']['efficiency']:.1f} samples/second")
        print(f"  Metrics saved to: {metrics_path}")
        
        # Verify metrics
        self.assertGreater(metrics['overall']['efficiency'], 0)
        self.assertEqual(
            metrics['data_splitting']['train_size'] + 
            metrics['data_splitting']['val_size'] + 
            metrics['data_splitting']['test_size'],
            len(stock_df)
        )


class TestRealWorldIntegration(unittest.TestCase):
    """Integration tests with real market data."""
    
    @classmethod
    def setUpClass(cls):
        """Check if we can run real-world tests."""
        cls.skip_real_world = False
        try:
            import yfinance as yf
            # Test connection
            test = yf.Ticker("SPY")
            data = test.history(period="1d")
            if data.empty:
                cls.skip_real_world = True
        except Exception as e:
            print(f"Skipping real-world tests: {e}")
            cls.skip_real_world = True
    
    def setUp(self):
        """Set up for each test."""
        if self.skip_real_world:
            self.skipTest("Real-world tests require internet connection")
        
        self.splitter = DataSplitter()
        self.recognizer = ChartPatternRecognizer()
    
    def test_sp500_components_pipeline(self):
        """Test pipeline with real S&P 500 components."""
        import yfinance as yf
        
        print("\n" + "="*60)
        print("Testing with Real S&P 500 Components")
        print("="*60)
        
        # Test with a few major stocks
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        stock_data = {}
        
        for ticker in tickers:
            print(f"\nFetching {ticker}...")
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            
            if not df.empty:
                stock_data[ticker] = df
                print(f"  {ticker}: {len(df)} days of data")
        
        if stock_data:
            # Split each stock's data
            train_data = {}
            test_data = {}
            
            for ticker, df in stock_data.items():
                train_df, val_df, test_df = self.splitter.chronological_split(
                    df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
                )
                train_data[ticker] = train_df
                test_data[ticker] = test_df
            
            # Batch detect patterns
            print("\nDetecting patterns in test data...")
            results = self.recognizer.batch_detect(test_data, confidence_threshold=0.3)
            
            # Analyze results
            print("\nPattern Detection Results:")
            total_patterns = 0
            for ticker, patterns in results.items():
                print(f"\n{ticker}: {len(patterns)} patterns")
                total_patterns += len(patterns)
                
                if patterns:
                    top_pattern = max(patterns, key=lambda p: p.confidence)
                    print(f"  Top pattern: {top_pattern.pattern_type} ({top_pattern.confidence:.1%})")
                    
                    # Check for breakouts
                    breakouts = [p for p in patterns if p.breakout_point is not None]
                    if breakouts:
                        print(f"  Breakouts detected: {len(breakouts)}")
            
            print(f"\nTotal patterns across {len(tickers)} stocks: {total_patterns}")
            print(f"Average patterns per stock: {total_patterns/len(tickers):.1f}")


def run_all_tests():
    """Run all integration tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestPatternRecognitionPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestRealWorldIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    return result


if __name__ == "__main__":
    print("="*60)
    print("INTEGRATION TEST SUITE")
    print("Pattern Recognition + Data Splitter Modules")
    print("="*60)
    run_all_tests()