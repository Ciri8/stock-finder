# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered trading bot system that combines multiple machine learning approaches including traditional ML, deep learning, reinforcement learning, and sentiment analysis for automated trading decisions. The codebase is currently in skeleton/template form with most implementation files empty but with a comprehensive dependency structure in place.

## Tech Stack

- **Python 3.x** - Primary language
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Deep Learning**: PyTorch 2.0.1, TensorFlow 2.13.0, Transformers
- **Reinforcement Learning**: stable-baselines3, gymnasium
- **Technical Analysis**: ta-lib, pandas-ta, vectorbt, backtrader
- **Data Sources**: yfinance, newsapi-python, beautifulsoup4
- **Visualization**: matplotlib, plotly, seaborn
- **Reporting**: reportlab, fpdf2

## Common Development Commands

```bash
# Install all dependencies
pip install -r requirements.txt

# Run the main application (once implemented)
python src/main.py

# Run tests
pytest tests/

# Launch Jupyter notebooks for research
jupyter notebook notebooks/

# Install TA-Lib (may require additional system dependencies)
# On Windows: Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# On Linux: sudo apt-get install ta-lib
# On Mac: brew install ta-lib
```

## Architecture and Structure

### Core Modules

1. **src/ai_models/** - Machine learning and AI components
   - `pattern_recognition.py` - Chart pattern detection using computer vision/ML
   - `price_prediction.py` - Time series forecasting models (LSTM, Transformer)
   - `risk_predictor.py` - Risk assessment and portfolio optimization
   - `rl_trader.py` - Reinforcement learning trading agent (PPO, A2C, SAC)
   - `sentiment_analysis.py` - NLP models for news/social sentiment

2. **src/data_pipeline/** - Data ingestion and processing
   - `fetcher.py` - Interfaces with yfinance, newsapi for real-time data
   - `preprocessor.py` - Feature engineering, normalization, missing data handling
   - `sp500_scraper.py` - S&P 500 specific data collection

3. **src/analysis/** - Market analysis components
   - `fundamental.py` - P/E ratios, earnings analysis, financial metrics
   - `market_regime.py` - Bull/bear market detection, volatility regimes
   - `signal_generator.py` - Combines multiple models to generate trading signals

4. **src/screening/** - Stock selection and filtering
   - `initial_filter.py` - Volume, market cap, liquidity filters
   - `technical_indicators.py` - RSI, MACD, Bollinger Bands calculations

5. **src/reporting/** - Output and visualization
   - `dashboard.py` - Real-time Plotly dashboard
   - `pdf_generator.py` - Performance reports using reportlab

### Data Flow

```
Raw Data (yfinance, news) → Data Pipeline → Preprocessing → 
→ Analysis & AI Models → Signal Generation → Risk Assessment → 
→ Trading Decisions → Reporting
```

### Key Design Patterns

- **Pipeline Pattern**: Data flows through sequential processing stages
- **Strategy Pattern**: Different AI models can be swapped for various market conditions
- **Observer Pattern**: Real-time monitoring and alert system
- **Factory Pattern**: Model creation based on configuration

## Implementation Priorities

When implementing features, follow this sequence:

1. **Data Pipeline First**: Implement fetcher.py and preprocessor.py to establish data flow
2. **Technical Indicators**: Build technical_indicators.py for basic signal generation
3. **Simple Models**: Start with price_prediction.py using basic LSTM
4. **Backtesting**: Use vectorbt or backtrader for strategy validation
5. **Risk Management**: Implement risk_predictor.py before live trading
6. **Reinforcement Learning**: rl_trader.py as advanced feature

## Key Considerations

### Data Management
- Use `data/raw/` for unprocessed market data
- Store cleaned data in `data/processed/`
- Cache frequently accessed data in `data/cache/`
- Save trained models in `data/models/` with versioning

### Model Development
- Use notebooks for experimentation before implementing in src/
- Implement proper train/validation/test splits
- Consider walk-forward analysis for time series validation
- Log all model metrics for comparison

### Risk Management
- Implement position sizing based on Kelly Criterion or risk parity
- Add stop-loss and take-profit mechanisms
- Monitor drawdown and implement circuit breakers
- Never exceed predefined risk limits

### Performance Optimization
- Use vectorized operations with numpy/pandas
- Implement caching for expensive computations
- Consider multiprocessing for parallel analysis
- Use generator patterns for large datasets

## Configuration

- **config/settings.py**: Application-wide settings (API keys, timeframes, risk parameters)
- **config/trading_rules.yaml**: Trading strategy rules and thresholds
- Use python-dotenv for sensitive configuration (API keys)

## Testing Strategy

```python
# Test data pipeline components
pytest tests/test_data_pipeline.py

# Test individual models
pytest tests/test_models.py

# Backtest strategies
python notebooks/03_backtesting.ipynb

# Integration tests
pytest tests/test_integration.py -v
```

## Common Tasks

### Adding a New Trading Strategy
1. Create new model in `src/ai_models/`
2. Add preprocessing in `src/data_pipeline/preprocessor.py`
3. Update signal generation in `src/analysis/signal_generator.py`
4. Backtest in `notebooks/03_backtesting.ipynb`
5. Add tests in `tests/`

### Training a Model
1. Prepare data using notebooks
2. Implement model class in appropriate module
3. Save trained model to `data/models/` with timestamp
4. Log performance metrics to `data/logs/`

### Running Backtests
1. Use vectorbt for quick vectorized backtesting
2. Use backtrader for more complex strategy logic
3. Generate reports in `data/reports/`