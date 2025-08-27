"""
Train Pattern Recognition Model with Real S&P 500 Data
This script implements semi-automatic labeling using technical indicators
and backtesting to validate patterns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yfinance as yf
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

from src.pattern_rec.pattern_recognition import (
    ChartPatternRecognizer, 
    ChartPatternDataset,
    PatternCNN
)
from src.screening.technical_indicators import TechnicalAnalyzer
from src.screening.breakout_filter import BreakoutFilter, FilterCriteria
from src.data_pipeline.sp500_scraper import SP500Scraper
from src.data_pipeline.fetcher import StockDataFetcher
from src.data_pipeline.preprocessor import DataPreprocessor
from src.utils.data_splitter import DataSplitter


class RealDataPatternLabeler:
    """
    Automatically labels chart patterns using technical analysis rules.
    This creates training data from real stocks.
    """
    
    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        
    def detect_bull_flag(self, df, window=60):
        """
        Detect bull flag: Strong uptrend → consolidation → breakout
        Returns confidence score (0-1)
        """
        if len(df) < window:
            return 0.0
            
        # Phase 1: Strong uptrend (first 40% of window)
        phase1_end = int(window * 0.4)
        phase1_return = (df['close'].iloc[phase1_end] - df['close'].iloc[0]) / df['close'].iloc[0]
        
        # Phase 2: Consolidation (middle 40%)
        phase2_start = phase1_end
        phase2_end = int(window * 0.8)
        phase2_volatility = df['close'].iloc[phase2_start:phase2_end].pct_change().std()
        phase2_range = (df['high'].iloc[phase2_start:phase2_end].max() - 
                       df['low'].iloc[phase2_start:phase2_end].min()) / df['close'].iloc[phase2_start]
        
        # Phase 3: Breakout (last 20%)
        breakout_level = df['high'].iloc[phase2_start:phase2_end].max()
        current_price = df['close'].iloc[-1]
        
        # Simple trend strength calculation
        sma_20 = df['close'].rolling(20).mean()
        if len(sma_20.dropna()) > 0:
            trend_strength = (df['close'].iloc[-1] - sma_20.iloc[-20]) / sma_20.iloc[-20] if len(sma_20) > 20 else 0
        else:
            trend_strength = 0
        
        # Scoring
        score = 0.0
        
        # Strong initial uptrend (>10% gain)
        if phase1_return > 0.10:
            score += 0.3
        elif phase1_return > 0.05:
            score += 0.15
            
        # Trend strength bonus
        if trend_strength > 0.1:
            score += 0.2
        elif trend_strength > 0.05:
            score += 0.1
            
        # Tight consolidation (low volatility, small range)
        if phase2_volatility < 0.02 and phase2_range < 0.05:
            score += 0.25
        elif phase2_volatility < 0.03 and phase2_range < 0.08:
            score += 0.1
            
        # Breakout above consolidation
        if current_price > breakout_level:
            score += 0.25
            
        return min(score, 1.0)
    
    def detect_cup_and_handle(self, df, window=60):
        """
        Detect cup and handle: U-shape recovery with small pullback
        """
        if len(df) < window:
            return 0.0
            
        prices = df['close'].values
        
        # Find the pattern phases
        first_quarter = window // 4
        mid_point = window // 2
        three_quarter = 3 * window // 4
        
        # Left rim
        left_high = prices[:first_quarter].max()
        
        # Bottom of cup
        bottom = prices[first_quarter:three_quarter].min()
        
        # Right rim  
        right_high = prices[three_quarter:three_quarter+10].max() if three_quarter+10 < len(prices) else prices[three_quarter:].max()
        
        # Handle (small pullback after right rim)
        if len(prices) > three_quarter + 10:
            handle_low = prices[three_quarter+10:].min()
        else:
            handle_low = prices[-1]
        
        # Calculate pattern score
        score = 0.0
        
        # U-shape: rims at similar height
        rim_similarity = 1 - abs(left_high - right_high) / left_high
        if rim_similarity > 0.95:
            score += 0.4
        elif rim_similarity > 0.90:
            score += 0.2
            
        # Depth: bottom is 10-30% below rims
        depth = (left_high - bottom) / left_high
        if 0.10 <= depth <= 0.30:
            score += 0.3
            
        # Handle: small pullback (less than 50% of cup depth)
        if handle_low > bottom + (right_high - bottom) * 0.5:
            score += 0.3
            
        return min(score, 1.0)
    
    def detect_head_and_shoulders(self, df, window=60):
        """
        Detect head and shoulders: Three peaks with middle highest
        """
        if len(df) < window:
            return 0.0
            
        highs = df['high'].values
        
        # Divide into sections for three peaks
        third = window // 3
        
        # Find peaks in each section
        left_peak = highs[:third].max()
        left_idx = highs[:third].argmax()
        
        center_peak = highs[third:2*third].max()
        center_idx = highs[third:2*third].argmax() + third
        
        right_peak = highs[2*third:].max()
        right_idx = highs[2*third:].argmax() + 2*third
        
        # Calculate pattern score
        score = 0.0
        
        # Center peak should be highest
        if center_peak > left_peak and center_peak > right_peak:
            score += 0.4
            
        # Shoulders should be similar height
        shoulder_similarity = 1 - abs(left_peak - right_peak) / left_peak
        if shoulder_similarity > 0.90:
            score += 0.3
            
        # Neckline break (price drops below shoulder lows)
        neckline = min(highs[left_idx:center_idx].min(), highs[center_idx:right_idx].min())
        if df['close'].iloc[-1] < neckline:
            score += 0.3
            
        return min(score, 1.0)
    
    def detect_double_bottom(self, df, window=60):
        """
        Detect double bottom: W-shape with two similar lows
        """
        if len(df) < window:
            return 0.0
            
        lows = df['low'].values
        
        # Find two bottoms
        first_half = window // 2
        
        first_bottom = lows[:first_half].min()
        first_idx = lows[:first_half].argmin()
        
        second_bottom = lows[first_half:].min()
        second_idx = lows[first_half:].argmin() + first_half
        
        # Peak between bottoms
        if first_idx < second_idx:
            middle_peak = df['high'].iloc[first_idx:second_idx].max()
        else:
            return 0.0
            
        score = 0.0
        
        # Bottoms at similar level (within 3%)
        bottom_similarity = 1 - abs(first_bottom - second_bottom) / first_bottom
        if bottom_similarity > 0.97:
            score += 0.4
        elif bottom_similarity > 0.95:
            score += 0.2
            
        # Middle peak creates the W (at least 5% above bottoms)
        peak_height = (middle_peak - first_bottom) / first_bottom
        if peak_height > 0.05:
            score += 0.3
            
        # Breakout above middle peak
        if df['close'].iloc[-1] > middle_peak:
            score += 0.3
            
        return min(score, 1.0)
    
    def label_pattern(self, df, window=60):
        """
        Analyze a price window and return the most likely pattern.
        Returns: (pattern_id, confidence)
        """
        if len(df) < window:
            return 0, 0.0  # no_pattern
            
        # Get last 'window' days
        df_window = df.tail(window).copy()
        
        # Detect all patterns
        patterns = {
            1: self.detect_bull_flag(df_window, window),        # bull_flag
            6: self.detect_cup_and_handle(df_window, window),   # cup_and_handle
            8: self.detect_head_and_shoulders(df_window, window), # head_shoulders
            10: self.detect_double_bottom(df_window, window),    # double_bottom
        }
        
        # Find best pattern
        best_pattern = max(patterns.items(), key=lambda x: x[1])
        pattern_id, confidence = best_pattern
        
        # If confidence too low, label as no_pattern
        if confidence < 0.4:  # Balanced threshold for quality patterns
            return 0, 1.0  # no_pattern with high confidence
            
        return pattern_id, confidence


def collect_sp500_data(limit=100):
    """Collect and filter S&P 500 stocks using all available components."""
    print("="*60)
    print("Collecting S&P 500 Data")
    print("="*60)
    
    # Initialize components
    scraper = SP500Scraper()
    fetcher = StockDataFetcher()
    breakout_filter = BreakoutFilter()
    preprocessor = DataPreprocessor()
    
    # Get S&P 500 list
    sp500_list = scraper.fetch_sp500_tickers()
    
    if limit:
        sp500_list = sp500_list[:limit]  # Use subset for faster training
    
    print(f"Processing {len(sp500_list)} stocks...")
    
    # Use BreakoutFilter with loose criteria for more training data
    criteria = FilterCriteria.loose()
    breakout_filter = BreakoutFilter(data_fetcher=fetcher, criteria=criteria)
    
    # Analyze stocks - increase max_results for more training data
    filtered_results = breakout_filter.analyze_stocks(
        tickers=sp500_list,
        fetch_data=True,
        max_results=100  # Get more stocks for training
    )
    
    # Process results
    filtered_stocks = {}
    
    for result in filtered_results:
        ticker = result.ticker
        
        # Get full 2-year history using fetcher (with caching)
        df = fetcher.fetch_stock_data(ticker, period="2y")
        
        if df is not None and len(df) > 250:  # At least 1 year of trading days
            # Preprocess the data for better features
            preprocessed = preprocessor.preprocess_stock(df)
            
            # Store preprocessed data (use features which includes OHLCV)
            filtered_stocks[ticker] = preprocessed.features
            
    print(f"Filtered to {len(filtered_stocks)} quality stocks")
    print(f"Filter criteria: {criteria.mode}")
    
    return filtered_stocks


def create_labeled_dataset(stock_data, window_size=60, stride=10):
    """
    Create labeled dataset from real stock data.
    Uses sliding window to extract multiple samples from each stock.
    """
    print("\n" + "="*60)
    print("Creating Labeled Dataset")
    print("="*60)
    
    recognizer = ChartPatternRecognizer()
    labeler = RealDataPatternLabeler()
    images = []
    labels = []
    metadata = []
    
    for ticker, df in tqdm(stock_data.items(), desc="Processing stocks"):
        # Add technical indicators using TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        
        # Calculate and add indicators to DataFrame
        df['rsi'] = analyzer.calculate_rsi(df)
        
        macd_df = analyzer.calculate_macd(df)
        if macd_df is not None and not macd_df.empty:
            for col in macd_df.columns:
                df[col] = macd_df[col]
        
        bb_df = analyzer.calculate_bollinger_bands(df)
        if bb_df is not None and not bb_df.empty:
            for col in bb_df.columns:
                df[col] = bb_df[col]
        
        ma_dict = analyzer.calculate_moving_averages(df)
        for name, series in ma_dict.items():
            df[name] = series
        
        vol_dict = analyzer.calculate_volume_indicators(df)
        for name, series in vol_dict.items():
            df[name] = series
        
        # Skip if not enough data
        if len(df) < window_size + 50:
            continue
            
        # Sliding window through historical data
        for i in range(0, len(df) - window_size - 20, stride):
            window = df.iloc[i:i+window_size].copy()
            
            # Look ahead 20 days to validate pattern
            future = df.iloc[i+window_size:i+window_size+20].copy()
            
            # Label the pattern
            pattern_id, confidence = labeler.label_pattern(window)
            
            # Validate with future performance
            future_return = (future['close'].iloc[-1] - window['close'].iloc[-1]) / window['close'].iloc[-1]
            
            # Adjust label based on actual outcome
            if pattern_id in [1, 6, 10]:  # Bullish patterns
                if future_return < -0.05:  # Pattern failed
                    pattern_id = 0  # Relabel as no_pattern
                    
            elif pattern_id == 8:  # Bearish pattern (head & shoulders)
                if future_return > 0.05:  # Pattern failed
                    pattern_id = 0  # Relabel as no_pattern
            
            # Only include high confidence labels
            if confidence > 0.6 or pattern_id == 0:
                # Convert to image
                image = recognizer.ohlcv_to_image(window, window_size=window_size)
                images.append(image)
                labels.append(pattern_id)
                metadata.append({
                    'ticker': ticker,
                    'date': window.index[-1].strftime('%Y-%m-%d'),
                    'pattern_id': pattern_id,
                    'confidence': confidence,
                    'future_return': future_return
                })
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Print statistics
    print(f"\nDataset created:")
    print(f"  Total samples: {len(images)}")
    print(f"  Pattern distribution:")
    
    pattern_names = {
        0: 'no_pattern',
        1: 'bull_flag', 
        6: 'cup_and_handle',
        8: 'head_shoulders',
        10: 'double_bottom'
    }
    
    for pattern_id, name in pattern_names.items():
        count = (labels == pattern_id).sum()
        print(f"    {name}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    return images, labels, metadata


def train_on_real_data(images, labels, metadata, epochs=30):
    """Train model on real labeled data."""
    print("\n" + "="*60)
    print("Training on Real S&P 500 Data")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model with correct number of patterns (5: no_pattern, bull_flag, cup_and_handle, head_shoulders, double_bottom)
    recognizer = ChartPatternRecognizer(device=str(device))
    # Reinitialize model with correct number of classes
    recognizer.model = PatternCNN(num_patterns=5, input_channels=3).to(device)
    recognizer.criterion = nn.CrossEntropyLoss()
    recognizer.optimizer = torch.optim.Adam(recognizer.model.parameters(), lr=0.0005)
    recognizer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(recognizer.optimizer, patience=5)
    splitter = DataSplitter()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.create_sequences_split(
        images, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Create datasets
    train_dataset = ChartPatternDataset(X_train, y_train)
    val_dataset = ChartPatternDataset(X_val, y_val)
    test_dataset = ChartPatternDataset(X_test, y_test)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    # Set learning rate before training
    recognizer.optimizer = torch.optim.Adam(recognizer.model.parameters(), lr=0.0005)
    history = recognizer.train_model(train_loader, val_loader, epochs=epochs, save_best=True)
    
    # Test accuracy
    recognizer.model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().to(device)
            
            outputs = recognizer.model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")
    
    # Save model and metadata
    model_path = 'data/models/real_data_pattern_model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    recognizer.save_model(model_path)
    
    # Save training metadata
    with open('data/models/training_metadata.json', 'w') as f:
        json.dump({
            'test_accuracy': test_accuracy,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'epochs': epochs,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, f, indent=2)
    
    print(f"\nModel saved to: {model_path}")
    
    return recognizer, test_accuracy


def backtest_patterns(recognizer, test_tickers=['AAPL', 'MSFT', 'NVDA', 'GOOGL']):
    """
    Backtest the trained model on recent data to see if patterns
    actually predict future price movements.
    """
    print("\n" + "="*60)
    print("Backtesting Pattern Predictions")
    print("="*60)
    
    fetcher = StockDataFetcher()
    preprocessor = DataPreprocessor()
    results = []
    
    for ticker in test_tickers:
        print(f"\nTesting {ticker}...")
        
        try:
            # Use fetcher for consistent data retrieval
            df = fetcher.fetch_stock_data(ticker, period="6mo")
            
            if df is not None:
                # Preprocess for better features
                preprocessed = preprocessor.preprocess_stock(df)
                df = preprocessed.features
            
            if len(df) >= 80:
                # Test on last 60 days, validate with next 20
                test_window = df.iloc[-80:-20]
                future_window = df.iloc[-20:]
                
                # Detect patterns
                patterns = recognizer.detect_patterns(test_window, confidence_threshold=0.6)
                
                if patterns:
                    for p in patterns:
                        # Calculate actual return
                        actual_return = (future_window['close'].iloc[-1] - test_window['close'].iloc[-1]) / test_window['close'].iloc[-1]
                        
                        print(f"  Pattern: {p.pattern_type} ({p.confidence:.1%} confidence)")
                        print(f"  20-day return: {actual_return:.2%}")
                        
                        # Evaluate prediction
                        if p.pattern_type in ['bull_flag', 'cup_and_handle', 'double_bottom']:
                            predicted_direction = 'UP'
                            correct = actual_return > 0
                        elif p.pattern_type in ['head_shoulders', 'bear_flag']:
                            predicted_direction = 'DOWN'
                            correct = actual_return < 0
                        else:
                            predicted_direction = 'NEUTRAL'
                            correct = abs(actual_return) < 0.03
                        
                        print(f"  Prediction: {predicted_direction} - {'✓ Correct' if correct else '✗ Wrong'}")
                        
                        results.append({
                            'ticker': ticker,
                            'pattern': p.pattern_type,
                            'confidence': p.confidence,
                            'actual_return': actual_return,
                            'correct': correct
                        })
                else:
                    print("  No patterns detected")
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    # Calculate overall accuracy
    if results:
        accuracy = sum(r['correct'] for r in results) / len(results)
        print(f"\nBacktest Accuracy: {accuracy:.1%}")
    
    return results


def main():
    """Main training pipeline using real S&P 500 data."""
    
    print("="*70)
    print(" REAL DATA PATTERN RECOGNITION TRAINING ")
    print("="*70)
    print("\nThis will:")
    print("1. Download S&P 500 stock data")
    print("2. Filter stocks using volume and price criteria")
    print("3. Automatically label patterns using technical analysis")
    print("4. Train CNN to recognize these patterns")
    print("5. Backtest on recent data")
    
    # Step 1: Collect S&P 500 data
    stock_data = collect_sp500_data(limit=200)  # Use more stocks for better accuracy
    
    # Step 2: Create labeled dataset with smaller stride for more samples
    images, labels, metadata = create_labeled_dataset(stock_data, window_size=60, stride=10)  # Reduced stride from 20 to 10
    
    if len(images) < 100:
        print("\n⚠️  Not enough training data. Try increasing the number of stocks or adjusting filters.")
        return
    
    # Step 3: Train model with more epochs for better accuracy
    recognizer, accuracy = train_on_real_data(images, labels, metadata, epochs=50)
    
    # Step 4: Backtest
    backtest_results = backtest_patterns(recognizer)
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE ")
    print("="*70)
    print(f"\nModel trained on {len(images)} real chart patterns")
    print(f"Test accuracy: {accuracy:.2f}%")
    print("\nTo use this model:")
    print("  recognizer = ChartPatternRecognizer(model_path='data/models/real_data_pattern_model.pth')")
    print("\nTo train with more data:")
    print("  Edit line 387: collect_sp500_data(limit=100)  # Use 100 stocks")


if __name__ == "__main__":
    main()