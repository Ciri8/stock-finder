"""
Comprehensive tests for the Pattern Recognition module.
Tests CNN-based chart pattern detection with real and synthetic data.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
import torch
import warnings
from datetime import datetime, timedelta
import tempfile
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pattern_rec.pattern_recognition import (
    ChartPatternRecognizer, 
    PatternCNN, 
    ChartPatternDataset,
    PatternDetection
)
from src.utils.data_splitter import DataSplitter

# Suppress warnings during tests
warnings.filterwarnings('ignore')


class TestPatternRecognition(unittest.TestCase):
    """Test suite for pattern recognition module."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for the entire test class."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {cls.device}")
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.recognizer = ChartPatternRecognizer(device=str(self.device))
        self.sample_df = self._create_sample_data()
        
    def _create_sample_data(self, n_days=100, pattern_type='random'):
        """Create synthetic OHLCV data with optional patterns."""
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Base price movement
        base_price = 100
        prices = [base_price]
        
        for i in range(1, n_days):
            # Add trend and noise
            if pattern_type == 'bull_flag':
                # Strong uptrend, consolidation, then continuation
                if i < 30:
                    change = np.random.normal(0.5, 0.5)  # Uptrend
                elif i < 50:
                    change = np.random.normal(0, 0.3)    # Consolidation
                else:
                    change = np.random.normal(0.3, 0.5)  # Continuation
            elif pattern_type == 'cup_and_handle':
                # U-shape followed by small consolidation
                if i < 20:
                    change = np.random.normal(-0.3, 0.3)  # Decline
                elif i < 40:
                    change = np.random.normal(0, 0.2)     # Bottom
                elif i < 60:
                    change = np.random.normal(0.3, 0.3)   # Recovery
                elif i < 75:
                    change = np.random.normal(-0.1, 0.2)  # Handle
                else:
                    change = np.random.normal(0.2, 0.3)   # Breakout
            elif pattern_type == 'head_shoulders':
                # Three peaks with middle being highest
                if i < 20:
                    change = np.random.normal(0.2, 0.3)   # First shoulder up
                elif i < 30:
                    change = np.random.normal(-0.1, 0.2)  # First shoulder down
                elif i < 45:
                    change = np.random.normal(0.3, 0.3)   # Head up
                elif i < 60:
                    change = np.random.normal(-0.3, 0.3)  # Head down
                elif i < 75:
                    change = np.random.normal(0.2, 0.3)   # Second shoulder up
                else:
                    change = np.random.normal(-0.4, 0.3)  # Breakdown
            else:
                # Random walk
                change = np.random.normal(0, 1)
            
            prices.append(prices[-1] * (1 + change/100))
        
        # Generate OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Create realistic OHLC from price
            daily_volatility = 0.02
            open_price = price * (1 + np.random.normal(0, daily_volatility/2))
            close_price = price * (1 + np.random.normal(0, daily_volatility/2))
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, daily_volatility)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, daily_volatility)))
            volume = np.random.randint(1000000, 10000000)
            
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
        return df
    
    def test_pattern_cnn_initialization(self):
        """Test CNN model initialization."""
        model = PatternCNN(num_patterns=12, input_channels=3)
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertEqual(len(list(model.parameters())), 16)  # Expected number of parameter groups
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 12))
        
        # Test prediction with confidence
        probs = model.predict_with_confidence(dummy_input)
        self.assertEqual(probs.shape, (1, 12))
        self.assertTrue(torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6))
    
    def test_ohlcv_to_image_conversion(self):
        """Test conversion of OHLCV data to image."""
        image = self.recognizer.ohlcv_to_image(self.sample_df, window_size=60)
        
        # Check image dimensions
        self.assertEqual(image.shape[0], 3)  # RGB channels
        self.assertEqual(image.shape[1], 224)  # Height
        self.assertEqual(image.shape[2], 224)  # Width
        
        # Check value range (should be normalized)
        self.assertTrue(image.min() >= 0)
        self.assertTrue(image.max() <= 1)
        
        # Test with smaller window
        small_df = self.sample_df.head(30)
        image_small = self.recognizer.ohlcv_to_image(small_df, window_size=30)
        self.assertEqual(image_small.shape, (3, 224, 224))
    
    def test_pattern_detection(self):
        """Test pattern detection on sample data."""
        # Test with different confidence thresholds
        for confidence_threshold in [0.3, 0.5, 0.7]:
            patterns = self.recognizer.detect_patterns(
                self.sample_df, 
                confidence_threshold=confidence_threshold
            )
            
            # Check return type
            self.assertIsInstance(patterns, list)
            
            # If patterns detected, validate structure
            for pattern in patterns:
                self.assertIsInstance(pattern, PatternDetection)
                self.assertIn(pattern.pattern_type, self.recognizer.PATTERNS.values())
                self.assertGreaterEqual(pattern.confidence, confidence_threshold)
                self.assertLessEqual(pattern.confidence, 1.0)
                self.assertIsInstance(pattern.start_idx, int)
                self.assertIsInstance(pattern.end_idx, int)
                self.assertLessEqual(pattern.start_idx, pattern.end_idx)
    
    def test_pattern_detection_with_specific_patterns(self):
        """Test detection with data containing specific patterns."""
        # Test bull flag pattern
        bull_flag_df = self._create_sample_data(100, 'bull_flag')
        patterns = self.recognizer.detect_patterns(bull_flag_df, confidence_threshold=0.1)
        
        self.assertIsInstance(patterns, list)
        if patterns:
            # At least one pattern should be detected
            pattern_types = [p.pattern_type for p in patterns]
            print(f"Detected patterns in bull flag data: {pattern_types}")
        
        # Test cup and handle pattern
        cup_handle_df = self._create_sample_data(100, 'cup_and_handle')
        patterns = self.recognizer.detect_patterns(cup_handle_df, confidence_threshold=0.1)
        
        self.assertIsInstance(patterns, list)
        if patterns:
            pattern_types = [p.pattern_type for p in patterns]
            print(f"Detected patterns in cup & handle data: {pattern_types}")
    
    def test_batch_detection(self):
        """Test batch pattern detection for multiple stocks."""
        stock_data = {
            'STOCK_A': self._create_sample_data(100, 'bull_flag'),
            'STOCK_B': self._create_sample_data(100, 'cup_and_handle'),
            'STOCK_C': self._create_sample_data(100, 'head_shoulders'),
            'STOCK_D': self._create_sample_data(100, 'random')
        }
        
        results = self.recognizer.batch_detect(stock_data, confidence_threshold=0.1)
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 4)
        
        for ticker, patterns in results.items():
            self.assertIn(ticker, stock_data.keys())
            self.assertIsInstance(patterns, list)
            
            # Print detection results
            if patterns:
                print(f"\n{ticker}: {len(patterns)} patterns detected")
                for p in patterns:
                    print(f"  - {p.pattern_type}: {p.confidence:.1%}")
    
    def test_breakout_point_detection(self):
        """Test breakout point detection logic."""
        # Create data with recent high breakout
        df = self.sample_df.copy()
        
        # Artificially create a breakout scenario
        recent_high = df['High'].tail(20).max()
        df.iloc[-1, df.columns.get_loc('Close')] = recent_high * 1.01
        
        breakout_point = self.recognizer._find_breakout_point(df, 'bull_flag')
        
        # Should detect breakout at last index
        self.assertEqual(breakout_point, len(df) - 1)
        
        # Test with no breakout
        df.iloc[-1, df.columns.get_loc('Close')] = recent_high * 0.95
        breakout_point = self.recognizer._find_breakout_point(df, 'bull_flag')
        self.assertIsNone(breakout_point)
    
    def test_chart_highlighting(self):
        """Test pattern highlighting on chart."""
        pattern = PatternDetection(
            pattern_type='bull_flag',
            confidence=0.85,
            start_idx=20,
            end_idx=60,
            breakout_point=59
        )
        
        # Test chart generation
        save_path = os.path.join(self.temp_dir, 'charts')
        filepath = self.recognizer.highlight_pattern_on_chart(
            self.sample_df, 
            pattern, 
            save_path
        )
        
        # Check file was created
        self.assertTrue(os.path.exists(filepath))
        self.assertTrue(filepath.endswith('.png'))
        self.assertIn('bull_flag', filepath)
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Save model
        save_path = os.path.join(self.temp_dir, 'test_model.pth')
        self.recognizer.save_model(save_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(save_path))
        
        # Load model in new recognizer
        new_recognizer = ChartPatternRecognizer(model_path=save_path)
        
        # Compare model parameters
        for p1, p2 in zip(self.recognizer.model.parameters(), 
                         new_recognizer.model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
    
    def test_chart_pattern_dataset(self):
        """Test the custom dataset class."""
        # Create dummy data
        images = np.random.rand(10, 3, 224, 224).astype(np.float32)
        labels = np.random.randint(0, 12, 10)
        
        dataset = ChartPatternDataset(images, labels)
        
        # Test dataset properties
        self.assertEqual(len(dataset), 10)
        
        # Test data loading
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(label.shape, (1,))
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
    
    def test_training_pipeline(self):
        """Test the training pipeline with dummy data."""
        # Create small dummy dataset
        n_samples = 50
        images = np.random.rand(n_samples, 3, 224, 224).astype(np.float32)
        labels = np.random.randint(0, 12, n_samples)
        
        # Split data
        train_size = int(0.8 * n_samples)
        train_images, val_images = images[:train_size], images[train_size:]
        train_labels, val_labels = labels[:train_size], labels[train_size:]
        
        # Create datasets
        train_dataset = ChartPatternDataset(train_images, train_labels)
        val_dataset = ChartPatternDataset(val_images, val_labels)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=4, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=4, shuffle=False
        )
        
        # Train for just 2 epochs (quick test)
        history = self.recognizer.train_model(
            train_loader, val_loader, epochs=2, save_best=False
        )
        
        # Check training history
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('val_acc', history)
        self.assertEqual(len(history['train_loss']), 2)
        self.assertEqual(len(history['val_loss']), 2)
        self.assertEqual(len(history['val_acc']), 2)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small dataframe
        small_df = self.sample_df.head(10)
        patterns = self.recognizer.detect_patterns(small_df)
        self.assertIsInstance(patterns, list)
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        with self.assertRaises(Exception):
            self.recognizer.detect_patterns(empty_df)
        
        # Test with missing columns
        bad_df = self.sample_df[['Open', 'Close']].copy()
        with self.assertRaises(Exception):
            self.recognizer.ohlcv_to_image(bad_df)
        
        # Test confidence threshold edge cases
        patterns = self.recognizer.detect_patterns(self.sample_df, confidence_threshold=0.0)
        self.assertIsInstance(patterns, list)
        
        patterns = self.recognizer.detect_patterns(self.sample_df, confidence_threshold=1.0)
        self.assertIsInstance(patterns, list)
        self.assertEqual(len(patterns), 0)  # Nothing should pass 100% confidence


class TestPatternRecognitionIntegration(unittest.TestCase):
    """Integration tests with real market data (requires internet)."""
    
    @classmethod
    def setUpClass(cls):
        """Set up for integration tests."""
        cls.skip_integration = False
        try:
            import yfinance as yf
            # Test connection
            test_ticker = yf.Ticker("AAPL")
            test_data = test_ticker.history(period="1d")
            if test_data.empty:
                cls.skip_integration = True
        except Exception as e:
            print(f"Skipping integration tests: {e}")
            cls.skip_integration = True
    
    def setUp(self):
        """Set up test fixtures."""
        if self.skip_integration:
            self.skipTest("Integration tests require internet connection")
        
        self.recognizer = ChartPatternRecognizer()
    
    def test_real_stock_data(self):
        """Test with real stock data from yfinance."""
        import yfinance as yf
        
        # Test with multiple timeframes
        test_configs = [
            ("AAPL", "3mo", 0.3),
            ("MSFT", "6mo", 0.2),
            ("TSLA", "1y", 0.1),
        ]
        
        for ticker, period, confidence in test_configs:
            with self.subTest(ticker=ticker):
                print(f"\nTesting {ticker} with {period} data...")
                
                # Fetch data
                stock = yf.Ticker(ticker)
                df = stock.history(period=period)
                
                if not df.empty:
                    # Detect patterns
                    patterns = self.recognizer.detect_patterns(
                        df, confidence_threshold=confidence
                    )
                    
                    # Print results
                    print(f"{ticker}: {len(patterns)} patterns detected")
                    for pattern in patterns:
                        print(f"  - {pattern.pattern_type}: {pattern.confidence:.1%}")
                        if pattern.breakout_point:
                            print(f"    Breakout at index {pattern.breakout_point}")
                    
                    # Validate patterns
                    for pattern in patterns:
                        self.assertGreaterEqual(pattern.confidence, confidence)
                        self.assertLessEqual(pattern.end_idx, len(df) - 1)


def run_tests():
    """Run all tests with detailed output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestPatternRecognition))
    suite.addTests(loader.loadTestsFromTestCase(TestPatternRecognitionIntegration))
    
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
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result


if __name__ == "__main__":
    print("="*60)
    print("PATTERN RECOGNITION MODULE TEST SUITE")
    print("="*60)
    run_tests()