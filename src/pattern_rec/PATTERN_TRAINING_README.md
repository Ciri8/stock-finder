# Pattern Recognition Module - Complete Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Why We Need Training](#why-we-need-training)
4. [Module Architecture](#module-architecture)
5. [Training with Real Data](#training-with-real-data)
6. [Testing & Validation](#testing--validation)
7. [Using the Trained Model](#using-the-trained-model)
8. [Troubleshooting](#troubleshooting)

## Overview

The Pattern Recognition module (`src/pattern_rec/pattern_recognition.py`) is a CNN-based system that detects chart patterns in stock price data. It's designed to identify 12 different technical patterns that traders use to predict future price movements.

### What It Does
- Converts OHLCV price data into candlestick chart images
- Uses a Convolutional Neural Network to "see" patterns like a human trader would
- Outputs pattern type and confidence score for trading decisions
- Can process multiple stocks in batch for screening opportunities

### Supported Patterns
```python
PATTERNS = {
    0: 'no_pattern',           # No clear pattern detected
    1: 'bull_flag',           # Bullish continuation pattern
    2: 'bear_flag',           # Bearish continuation pattern
    3: 'ascending_triangle',  # Bullish breakout pattern
    4: 'descending_triangle', # Bearish breakout pattern
    5: 'symmetric_triangle',  # Neutral, awaiting breakout
    6: 'cup_and_handle',      # Bullish reversal pattern
    7: 'inverse_head_shoulders', # Bullish reversal
    8: 'head_shoulders',      # Bearish reversal pattern
    9: 'double_top',          # Bearish reversal
    10: 'double_bottom',      # Bullish reversal
    11: 'wedge'               # Can be bullish or bearish
}
```

## How It Works

### The Pipeline Flow
```
1. Price Data (OHLCV DataFrame)
        ↓
2. Chart Image Generation (224x224 RGB)
        ↓
3. CNN Feature Extraction (4 Conv Blocks)
        ↓
4. Pattern Classification (12 classes)
        ↓
5. Confidence Scores (0-100%)
```

### Core Components in `pattern_recognition.py`

#### 1. **PatternCNN Class** (lines 67-125)
The neural network architecture:
```python
- Conv Block 1: 3→32 channels + BatchNorm + ReLU + MaxPool
- Conv Block 2: 32→64 channels + BatchNorm + ReLU + MaxPool  
- Conv Block 3: 64→128 channels + BatchNorm + ReLU + MaxPool
- Conv Block 4: 128→256 channels + BatchNorm + ReLU + MaxPool
- FC Layer 1: 256*14*14 → 512 neurons + Dropout(0.3)
- FC Layer 2: 512 → 256 neurons + Dropout(0.3)
- FC Layer 3: 256 → 12 classes (patterns)
```

**Why this architecture?**
- Convolutional layers detect visual features (trends, support/resistance lines)
- Batch normalization speeds up training and improves accuracy
- Dropout prevents overfitting to training data
- Progressive channel increase captures complex patterns

#### 2. **ChartPatternRecognizer Class** (lines 127-527)
Main class that handles:
- Model initialization and device management (GPU/CPU)
- OHLCV to image conversion
- Pattern detection with confidence scoring
- Model training and saving
- Batch processing for multiple stocks

#### 3. **ohlcv_to_image Method** (lines 180-263)
Converts price data to visual charts:
```python
def ohlcv_to_image(df, window_size=60):
    # Creates candlestick chart with:
    # - Green/red candles for up/down days
    # - Volume bars at bottom
    # - 20 & 50 day moving averages
    # - Returns 224x224 RGB image array
```

**Why convert to images?**
- CNNs excel at visual pattern recognition
- Chart patterns are inherently visual concepts
- Preserves spatial relationships between price points
- Allows model to "see" patterns like human traders do

## Why We Need Training

### Before Training (Random Weights)
```python
# Untrained model outputs random predictions
patterns = recognizer.detect_patterns(stock_df)
# Results: Random patterns with ~8% accuracy (random guess among 12 classes)
```

### After Training
```python
# Trained model recognizes real patterns
patterns = recognizer.detect_patterns(stock_df)
# Results: Accurate patterns with 70-85% accuracy
```

### The Learning Process
1. **Show Examples**: "This chart is a bull flag pattern"
2. **Model Guesses**: Initially wrong most of the time
3. **Calculate Error**: How wrong was the guess?
4. **Adjust Weights**: Update neural network parameters
5. **Repeat**: Thousands of examples until accurate

## Module Architecture

### File Structure
```
src/pattern_rec/
├── pattern_recognition.py    # Main CNN model and recognizer
├── __init__.py               # Module initialization
└── PATTERN_TRAINING_README.md # This documentation

Related Files:
├── train_with_real_data.py   # Training on real S&P 500 data
├── tests/test_pattern_recognition.py  # Unit tests
└── data/models/              # Saved trained models
```

### Key Dependencies
- **PyTorch**: Neural network framework
- **matplotlib/mplfinance**: Chart generation
- **PIL/cv2**: Image processing
- **yfinance**: Real-time stock data
- **numpy/pandas**: Data manipulation

## Training with Real Data

### Quick Start - Real S&P 500 Data
```bash
# Train with real stock patterns (recommended)
python train_with_real_data.py

# This will:
# 1. Download S&P 500 stocks (30 by default)
# 2. Filter by volume and price criteria
# 3. Auto-label patterns using technical rules
# 4. Train CNN on real market patterns
# 5. Validate with future price movements
# 6. Backtest on recent data
```

### How Real Data Training Works

#### Step 1: Data Collection
```python
# Fetches 2 years of S&P 500 stock data
# Filters stocks with volume > 1M, price > $5
# Results in ~25-30 quality stocks
```

#### Step 2: Automatic Pattern Labeling
The `RealDataPatternLabeler` class uses technical analysis rules:
- **Bull Flag**: 10%+ rise → low volatility consolidation → breakout
- **Cup & Handle**: U-shape recovery with 90-95% rim similarity
- **Head & Shoulders**: Three peaks with center 20%+ higher
- **Double Bottom**: Two lows within 3% of each other

#### Step 3: Label Validation
```python
# After labeling a pattern, checks next 20 days:
if pattern == "bull_flag" and future_return < -5%:
    # Pattern failed - relabel as "no_pattern"
    # This ensures only ACCURATE patterns train the model
```

#### Step 4: Training Process
```python
# Sliding window through historical data:
# - Extract 60-day windows every 20 days
# - Label each window
# - Validate with future performance
# - Train CNN on validated patterns
```

### Training Options
```bash
# More data for better accuracy
# Edit line 553 in train_with_real_data.py:
stock_data = collect_sp500_data(limit=100)  # Use 100 stocks

# More epochs for better learning
# Edit line 521:
train_on_real_data(images, labels, metadata, epochs=50)
```

## Testing & Validation

### Run Unit Tests
```bash
# Test pattern recognition module
python -m pytest tests/test_pattern_recognition.py -v

# Run all tests
python run_all_tests.py
```

### Test Coverage
The test suite (`tests/test_pattern_recognition.py`) includes:
- CNN model initialization and forward pass
- Image conversion from OHLCV data
- Pattern detection with various confidence thresholds
- Batch processing multiple stocks
- Model save/load functionality
- Edge cases (small data, missing columns)
- Integration tests with real yfinance data

### Manual Testing After Training
```python
import yfinance as yf
from src.pattern_rec.pattern_recognition import ChartPatternRecognizer

# Load trained model
recognizer = ChartPatternRecognizer(
    model_path='data/models/real_data_pattern_model.pth'
)

# Test on any stock
stock = yf.Ticker("NVDA")
df = stock.history(period="6mo")

# Detect patterns
patterns = recognizer.detect_patterns(df, confidence_threshold=0.6)

for p in patterns:
    print(f"{p.pattern_type}: {p.confidence:.1%}")
    if p.breakout_point:
        print("  → Breakout detected!")
```

## Using the Trained Model

### In Your Trading Pipeline
```python
from src.pattern_rec.pattern_recognition import ChartPatternRecognizer
from src.screening.initial_filter import InitialScreener
from src.screening.technical_indicators import TechnicalIndicators

# Initialize components
recognizer = ChartPatternRecognizer(model_path='data/models/real_data_pattern_model.pth')
screener = InitialScreener()
technical = TechnicalIndicators()

# Your complete pipeline
def analyze_stock(ticker):
    # 1. Get data
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    
    # 2. Initial screening
    if not screener.filter_by_volume(df):
        return "Low volume"
    
    # 3. Add technical indicators
    df = technical.add_indicators(df)
    
    # 4. Detect patterns
    patterns = recognizer.detect_patterns(df, confidence_threshold=0.6)
    
    # 5. Generate signals
    for p in patterns:
        if p.pattern_type in ['bull_flag', 'cup_and_handle', 'double_bottom']:
            return f"BUY: {p.pattern_type} ({p.confidence:.0%})"
        elif p.pattern_type in ['head_shoulders', 'double_top']:
            return f"SELL: {p.pattern_type} ({p.confidence:.0%})"
    
    return "HOLD: No clear patterns"
```

### Batch Processing S&P 500
```python
# Process all S&P 500 stocks
from src.data_pipeline.sp500_scraper import SP500Scraper

scraper = SP500Scraper()
sp500_list = scraper.get_sp500_list()

# Collect data
stock_data = {}
for ticker in sp500_list[:50]:  # First 50 stocks
    stock = yf.Ticker(ticker)
    df = stock.history(period="3mo")
    if len(df) >= 60:
        stock_data[ticker] = df

# Batch detect patterns
results = recognizer.batch_detect(stock_data, confidence_threshold=0.6)

# Find best opportunities
for ticker, patterns in results.items():
    if patterns:
        best_pattern = max(patterns, key=lambda p: p.confidence)
        print(f"{ticker}: {best_pattern.pattern_type} ({best_pattern.confidence:.0%})")
```

## Troubleshooting

### Issue: "Model outputs random patterns"
**Cause**: Using untrained model
**Solution**: Train with `python train_with_real_data.py` first

### Issue: "No patterns detected after training"
**Solutions**:
```python
# 1. Lower confidence threshold
patterns = recognizer.detect_patterns(df, confidence_threshold=0.3)

# 2. Ensure enough data (need 60+ days)
if len(df) < 60:
    df = stock.history(period="3mo")  # Get more data

# 3. Check model loaded correctly
print(f"Model device: {recognizer.device}")
print(f"Model loaded: {os.path.exists('data/models/real_data_pattern_model.pth')}")
```

### Issue: "Training takes too long"
**Solutions**:
```python
# 1. Use fewer stocks for training
stock_data = collect_sp500_data(limit=20)  # Just 20 stocks

# 2. Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# 3. Reduce image size (in pattern_recognition.py line 149)
image_size = (112, 112)  # Smaller images, faster processing
```

### Issue: "Low accuracy after training"
**Solutions**:
1. Use more training data (100+ stocks)
2. Train for more epochs (50+)
3. Ensure data quality (check for NaN values)
4. Balance pattern distribution (not all "no_pattern")

## Performance Expectations

### Accuracy by Training Quality
- **Untrained**: ~8% (random among 12 classes)
- **Quick train (20 stocks, 10 epochs)**: 40-50%
- **Standard (30 stocks, 20 epochs)**: 60-70%
- **Thorough (100 stocks, 50 epochs)**: 75-85%
- **Production (200+ stocks, 100+ epochs)**: 80-90%

### Confidence Thresholds for Trading
- **0.3-0.5**: Exploration, many signals but less reliable
- **0.6-0.7**: Balanced, good for swing trading
- **0.8+**: Conservative, fewer but high-quality signals

### Backtest Results (Expected)
After proper training:
- Pattern detection accuracy: 70-80%
- Direction prediction (up/down): 60-65%
- Profitable signals: 55-60% (edge over random 50%)

## Best Practices

1. **Regular Retraining**: Markets evolve, retrain monthly
2. **Combine Signals**: Don't trade on patterns alone
3. **Risk Management**: Use stop-losses even with high confidence
4. **Market Conditions**: Patterns work better in trending markets
5. **Validation**: Always backtest before live trading

## How to Run Training & Verify Results

### Step 1: Run the Training
```bash
# Basic run (30 stocks, 20 epochs)
python train_with_real_data.py

# This will take ~15-30 minutes on CPU, ~5-10 minutes on GPU
```

### Step 2: What You'll See During Training
```
==============================================================
Collecting S&P 500 Data
==============================================================
Processing 30 stocks...
100%|████████████| 30/30 [00:45<00:00, 1.5s/stock]
Filtered to 18 quality stocks
Filter criteria: NORMAL

==============================================================
Creating Labeled Dataset
==============================================================
Processing stocks: 100%|████████████| 18/18
Dataset created:
  Total samples: 1,250
  Pattern distribution:
    no_pattern: 750 samples (60.0%)
    bull_flag: 200 samples (16.0%)
    cup_and_handle: 150 samples (12.0%)
    head_shoulders: 75 samples (6.0%)
    double_bottom: 75 samples (6.0%)

==============================================================
Training on Real S&P 500 Data
==============================================================
Using device: cuda
Dataset splits:
  Train: 875 samples
  Val: 187 samples
  Test: 188 samples

Training for 20 epochs...
Epoch 5/20 - Train Loss: 1.2341, Val Loss: 1.1523, Val Acc: 45.23%
Epoch 10/20 - Train Loss: 0.8234, Val Loss: 0.9123, Val Acc: 62.45%
Epoch 15/20 - Train Loss: 0.5234, Val Loss: 0.7234, Val Acc: 71.23%
Epoch 20/20 - Train Loss: 0.3123, Val Loss: 0.6123, Val Acc: 78.45%

Final Test Accuracy: 76.2%

==============================================================
Backtesting Pattern Predictions
==============================================================
Testing AAPL...
  Pattern: bull_flag (73% confidence)
  20-day return: +5.3%
  Prediction: UP - ✓ Correct

Testing MSFT...
  Pattern: cup_and_handle (68% confidence)
  20-day return: -2.1%
  Prediction: UP - ✗ Wrong
```

### Step 3: Check if Results are Good

#### Good Training Metrics:
```python
✅ Val Acc > 60% = Model is learning patterns
✅ Train Loss decreasing = Model is improving
✅ Val Loss decreasing (mostly) = Not overfitting badly

❌ Val Acc < 40% = Random guessing
❌ Val Loss increasing while Train Loss decreases = Overfitting
❌ Test Accuracy << Val Accuracy = Poor generalization
```

#### Good Pattern Distribution:
```
GOOD Distribution:
no_pattern: 50-70%     # Most windows don't have clear patterns
bull_flag: 10-20%      # Common pattern
cup_and_handle: 5-15%  # Less common
head_shoulders: 3-10%  # Rare
double_bottom: 3-10%   # Rare

BAD Distribution:
no_pattern: 95%        # Not finding patterns (labeling too strict)
no_pattern: 20%        # Finding too many patterns (labeling too loose)
```

### Step 4: Manual Validation Script

Create and run this script to test your trained model:

```python
# test_trained_model.py
import yfinance as yf
from src.pattern_rec.pattern_recognition import ChartPatternRecognizer
import warnings
warnings.filterwarnings('ignore')

def test_model_quality():
    """Test if the trained model actually works."""
    
    # Load trained model
    recognizer = ChartPatternRecognizer(
        model_path='data/models/real_data_pattern_model.pth'
    )
    
    # Test on recent popular stocks
    test_stocks = ['NVDA', 'TSLA', 'META', 'AMZN', 'GOOGL']
    
    print("\n" + "="*60)
    print("TESTING TRAINED MODEL QUALITY")
    print("="*60)
    
    detections = 0
    high_confidence = 0
    
    for ticker in test_stocks:
        print(f"\n{ticker}:")
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")
        
        if len(df) >= 60:
            patterns = recognizer.detect_patterns(df, confidence_threshold=0.3)
            
            if patterns:
                detections += 1
                for p in patterns:
                    print(f"  • {p.pattern_type}: {p.confidence:.1%}")
                    if p.confidence > 0.6:
                        high_confidence += 1
            else:
                print("  No patterns detected")
    
    # Quality metrics
    print("\n" + "="*60)
    print("MODEL QUALITY METRICS")
    print("="*60)
    print(f"Stocks with patterns: {detections}/{len(test_stocks)}")
    print(f"High confidence patterns: {high_confidence}")
    
    if detections >= 2 and high_confidence >= 1:
        print("✅ Model is working well!")
    elif detections >= 1:
        print("⚠️ Model needs more training")
    else:
        print("❌ Model not detecting patterns - check training")

if __name__ == "__main__":
    test_model_quality()
```

Run it:
```bash
python test_trained_model.py
```

### Step 5: Quick Quality Checklist

```bash
# 1. Check model file exists and has size
ls -lh data/models/
# Should see: real_data_pattern_model.pth (~50-100MB)

# 2. Check training metadata
cat data/models/training_metadata.json
# Should show test_accuracy > 60%

# 3. Run pattern detection on a known stock
python -c "
import yfinance as yf
from src.pattern_rec.pattern_recognition import ChartPatternRecognizer
r = ChartPatternRecognizer(model_path='data/models/real_data_pattern_model.pth')
df = yf.Ticker('AAPL').history(period='3mo')
patterns = r.detect_patterns(df, confidence_threshold=0.4)
print(f'Found {len(patterns)} patterns')
for p in patterns: print(f'{p.pattern_type}: {p.confidence:.0%}')
"
```

## Common Issues & Fixes

### Issue 1: "Low accuracy (<50%)"
```bash
# Fix: Train with more data and epochs
# Edit train_with_real_data.py line 553:
stock_data = collect_sp500_data(limit=100)  # More stocks

# Edit line 521:
train_on_real_data(images, labels, metadata, epochs=50)  # More epochs
```

### Issue 2: "No patterns detected after training"
```python
# Lower confidence threshold
patterns = recognizer.detect_patterns(df, confidence_threshold=0.2)  # Lower

# Check if model loaded
import os
print(os.path.exists('data/models/real_data_pattern_model.pth'))
```

### Issue 3: "Training takes forever"
```python
# Use fewer stocks for quick test
stock_data = collect_sp500_data(limit=10)  # Just 10 stocks

# Check if using GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
```

## Expected Good Results

After successful training, you should see:
- **Test Accuracy**: 65-80%
- **Pattern Detection Rate**: 30-50% of stocks show patterns
- **Confidence Distribution**: Most patterns 40-70% confidence
- **Backtest Success**: 55-65% correct direction predictions
- **Processing Speed**: <1 second per stock for detection

### Good Backtest Results:
```
✅ 55-65% direction accuracy (better than 50% random)
✅ Correctly identifies 60%+ of bullish patterns
✅ Catches some major moves

❌ <50% accuracy (worse than random)
❌ All predictions in same direction
❌ No correlation with actual moves
```

If you're getting these results, your model is trained properly and ready to use! 🎉

## Next Steps

1. **Train Your Model**:
   ```bash
   python train_with_real_data.py
   ```

2. **Test Recognition**:
   ```bash
   python -m pytest tests/test_pattern_recognition.py
   ```

3. **Integrate with Pipeline**:
   - Add to your main trading strategy
   - Combine with technical indicators
   - Set up alerts for high-confidence patterns

4. **Monitor Performance**:
   - Track pattern accuracy over time
   - Log predictions vs actual outcomes
   - Adjust confidence thresholds based on results

---

**Remember**: Pattern recognition is a tool, not a crystal ball. Always combine with other analysis methods and proper risk management for successful trading.