# Test Documentation - AI Trading Bot

## Overview
This document provides a comprehensive overview of all test files in the AI Trading Bot project, what components they test, and their coverage status.

## Test Files Summary

| Test File | Tests Module(s) | Status | Coverage |
|-----------|----------------|--------|----------|
| `test_pattern_recognition.py` | `src.pattern_rec.pattern_recognition` | ✅ Fixed | Good |
| `test_data_splitter.py` | `src.utils.data_splitter` | ✅ Working | Good |
| `test_integration.py` | Multiple modules (integration) | ✅ Fixed | Good |
| `test_models.py` | Test orchestrator | ✅ Working | N/A |
| `test_breakout.py` | `src.screening.breakout_filter` | ✅ Fixed | Good |

---

## Detailed Test Coverage

### 1. `test_pattern_recognition.py`
**Purpose:** Tests the CNN-based chart pattern recognition system

**Module Tested:** 
- ✅ `src.pattern_rec.pattern_recognition` (CORRECT PATH)

**Test Classes:**
- `TestPatternRecognition` - Unit tests for pattern recognition
- `TestPatternRecognitionIntegration` - Integration tests with real data

**What It Tests:**
- CNN model architecture (`PatternCNN`)
- Chart pattern dataset creation (`ChartPatternDataset`)
- Pattern detection logic (`PatternDetection`)
- Image conversion from OHLCV data
- Model training and evaluation
- Pattern confidence scoring
- Model save/load functionality

**Coverage:** ✅ Comprehensive - All major functions tested

---

### 2. `test_data_splitter.py`
**Purpose:** Tests various data splitting strategies for time series and ML models

**Module Tested:** 
- ✅ `src.utils.data_splitter.DataSplitter`

**Test Classes:**
- `TestDataSplitter` - Unit tests for data splitting
- `TestDataSplitterIntegration` - Integration tests with various data types

**What It Tests:**
- Chronological splitting (time series respect)
- Random splitting with stratification
- Walk-forward validation splits
- Sequence data splitting (for LSTM/RNN)
- Train/validation/test split ratios
- Data leakage prevention
- Split reproducibility with seeds
- Edge cases (small datasets, imbalanced classes)

**Coverage:** ✅ Comprehensive

---

### 3. `test_integration.py`
**Purpose:** End-to-end integration testing of the ML pipeline

**Modules Tested:**
- ✅ `src.pattern_rec.pattern_recognition`
- ✅ `src.utils.data_splitter`

**Test Classes:**
- `TestPatternRecognitionPipeline` - Full pipeline testing
- `TestRealWorldIntegration` - Tests with realistic market scenarios

**What It Tests:**
- Complete workflow from data → splitting → training → prediction
- Multi-stock pattern detection
- Model performance on unseen data
- Pipeline error handling
- Memory efficiency with large datasets
- Cross-validation with time series

**Coverage:** ✅ Comprehensive integration testing

---

### 4. `test_models.py`
**Purpose:** Main test orchestrator that runs all test suites

**Function:** 
- Imports all test classes from other test files
- Creates unified test suite
- Provides single entry point for running all tests

**What It Does:**
- Aggregates all test classes
- Configures test runner with verbosity
- Ensures all tests run in correct order

---

### 5. `test_breakout.py`
**Purpose:** Tests the breakout detection and filtering system

**Module Tested:**
- ✅ `src.screening.breakout_filter`
- ✅ `src.data_pipeline.fetcher`
- ✅ `src.data_pipeline.sp500_scraper`

**Test Functions:**
- `test_filter_modes()` - Tests STRICT/NORMAL/LOOSE filtering modes
- `test_specific_stocks()` - Tests breakout detection on specific tickers
- `test_criteria_comparison()` - Compares different filter criteria settings

**What It Tests:**
- Breakout detection across different market conditions
- Filter criteria modes (strict, normal, loose)
- Volume surge detection
- Price momentum calculations
- Quality grading system (A+, A, B+, etc.)
- Integration with S&P 500 stock screening
- Performance with large stock lists (500+ stocks)
- Specific stock analysis for known tickers

**Coverage:** ✅ Comprehensive functional testing

---

## Components WITHOUT Tests

The following implemented components currently have NO test coverage:

### 🔴 Missing Tests - Critical Components

1. **`src.screening.technical_indicators.py`**
   - `TechnicalIndicators` class
   - `TechnicalAnalyzer` class
   - RSI, MACD, Bollinger Bands calculations
   - Signal generation logic

2. **`src.data_pipeline.preprocessor.py`**
   - `DataPreprocessor` class
   - `PreprocessedData` dataclass
   - Feature engineering pipeline
   - Missing data handling
   - Outlier detection
   - Data normalization

4. **`src.data_pipeline.fetcher.py`**
   - `StockDataFetcher` class
   - Caching mechanisms
   - API error handling
   - Rate limiting
   - Data validation

5. **`src.data_pipeline.sp500_scraper.py`**
   - `SP500Scraper` class
   - Web scraping reliability
   - Data freshness checks
   - Error recovery

### ✅ Test Files Status

1. **All Import Paths Fixed:**
   - Tests now correctly import from `src.pattern_rec`
   - All test files should run without import errors

2. **Missing Integration Tests:**
   - No tests for `DataPreprocessor` + `BreakoutFilter` integration
   - No tests for `StockDataFetcher` + `TechnicalIndicators` integration
   - No tests for the complete `train_with_real_data.py` workflow

---

## Recommended Test Additions

### ✅ Priority 1 - COMPLETED
All existing test files have been fixed with correct import paths.

### Priority 2 - Add Missing Unit Tests

1. **`test_technical_indicators.py`**
   - Test RSI calculation accuracy
   - Test MACD signals
   - Test Bollinger Band calculations
   - Test edge cases (insufficient data, NaN handling)

2. **`test_preprocessor.py`**
   - Test feature engineering
   - Test missing data handling
   - Test normalization methods
   - Test outlier detection

3. **`test_fetcher.py`**
   - Test caching behavior
   - Test API error handling
   - Test data validation
   - Mock yfinance calls for reliability

### Priority 3 - Add Integration Tests

1. **`test_full_pipeline.py`**
   - Test complete `train_with_real_data.py` workflow
   - Test all components working together
   - Test with real S&P 500 data (small subset)
   - Test model persistence and loading

---

## How to Run Tests

### Run All Tests
```bash
# Using pytest (recommended)
pytest tests/ -v

# Using unittest
python -m unittest discover tests/

# Run specific test file
pytest tests/test_data_splitter.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Run Individual Test Classes
```python
# Run from project root
python tests/test_models.py

# Run specific test class
python -m unittest tests.test_data_splitter.TestDataSplitter
```

---

## Test Coverage Summary

### Current Coverage Status
- ✅ **Good Coverage:** `DataSplitter`, `PatternRecognition`, `BreakoutFilter`
- 🔴 **No Coverage:** `TechnicalIndicators`, `DataPreprocessor`, `StockDataFetcher`, `SP500Scraper`

### Overall Test Health: 5/10
- 3 components fully tested (DataSplitter, PatternRecognition, BreakoutFilter)
- Good integration test coverage
- 4+ components still need unit tests

### Recommended Actions
1. ~~**Immediate:** Fix import paths in existing tests~~ ✅ DONE
2. ~~**Immediate:** Move test_breakout.py to tests folder~~ ✅ DONE
3. **High Priority:** Add tests for `TechnicalIndicators`
4. **Medium Priority:** Add tests for data pipeline components
5. **Low Priority:** Add more integration tests

---

## Testing Best Practices for This Project

1. **Mock External APIs:** Use `unittest.mock` to mock yfinance calls
2. **Use Fixtures:** Create reusable test data fixtures
3. **Test Edge Cases:** Empty dataframes, missing values, extreme market conditions
4. **Time Series Awareness:** Ensure tests respect temporal ordering
5. **Performance Tests:** Add tests for large dataset handling
6. **Deterministic Tests:** Use fixed random seeds for reproducibility

---

## Notes for Developers

- All test files should be prefixed with `test_`
- Test classes should inherit from `unittest.TestCase`
- Use descriptive test method names: `test_<what>_<condition>_<expected>`
- Add docstrings explaining what each test validates
- Group related tests in the same test class
- Use `setUp()` and `tearDown()` for test fixtures
- Clean up temporary files/directories after tests