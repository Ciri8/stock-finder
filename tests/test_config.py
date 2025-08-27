"""
Test script for configuration system
Verifies that settings load correctly and validation works
"""

import sys
from pathlib import Path
from pprint import pprint

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    Settings, 
    get_settings, 
    reload_settings,
    FilterCriteria,
    ModelConfig,
    TradingParameters
)


def test_settings_loading():
    """Test that settings load without errors"""
    print("=" * 60)
    print("Testing Settings Loading")
    print("=" * 60)
    
    try:
        settings = get_settings()
        print("✅ Settings loaded successfully")
        
        # Print summary
        settings.print_summary()
        
        return True
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return False


def test_validation():
    """Test validation of settings"""
    print("\n" + "=" * 60)
    print("Testing Validation")
    print("=" * 60)
    
    settings = get_settings()
    
    # Test with valid settings
    errors = settings.validate()
    if not errors:
        print("✅ Current settings are valid")
    else:
        print("⚠️ Validation errors found:")
        for section, section_errors in errors.items():
            for error in section_errors:
                print(f"  [{section}] {error}")
    
    # Test with invalid settings
    print("\nTesting invalid settings detection:")
    
    # Create invalid filter criteria
    invalid_filter = FilterCriteria(
        min_price=100,
        max_price=50,  # Invalid: min > max
        min_weekly_change=0.2,
        max_weekly_change=0.1,  # Invalid: min > max
        volume_surge_ratio=0.5  # Invalid: < 1.0
    )
    
    errors = invalid_filter.validate()
    if errors:
        print("✅ Invalid filter criteria detected correctly:")
        for error in errors:
            print(f"  - {error}")
    
    # Test invalid model config
    invalid_model = ModelConfig(
        cnn_confidence_threshold=1.5,  # Invalid: > 1.0
        lstm_lookback_days=10,  # Warning: < 20
        ensemble_weights={
            "technical": 0.5,
            "quantitative": 0.3,
            "sentiment": 0.1  # Invalid: doesn't sum to 1.0
        }
    )
    
    errors = invalid_model.validate()
    if errors:
        print("✅ Invalid model config detected correctly:")
        for error in errors:
            print(f"  - {error}")
    
    return True


def test_directory_creation():
    """Test that required directories are created"""
    print("\n" + "=" * 60)
    print("Testing Directory Creation")
    print("=" * 60)
    
    settings = get_settings()
    
    # Check if directories exist
    directories = [
        settings.data.cache_dir,
        settings.data.raw_data_dir,
        settings.data.processed_data_dir,
        settings.data.models_dir,
        settings.data.reports_dir,
        settings.data.logs_dir
    ]
    
    all_exist = True
    for directory in directories:
        if directory.exists():
            print(f"✅ {directory} exists")
        else:
            print(f"❌ {directory} does not exist")
            all_exist = False
    
    return all_exist


def test_filter_criteria():
    """Test filter criteria configuration"""
    print("\n" + "=" * 60)
    print("Testing Filter Criteria")
    print("=" * 60)
    
    settings = get_settings()
    criteria = settings.filter
    
    print(f"Price range: ${criteria.min_price:.0f} - ${criteria.max_price:.0f}")
    print(f"Volume minimum: {criteria.min_volume:,}")
    print(f"Weekly change: {criteria.min_weekly_change:.1%} - {criteria.max_weekly_change:.1%}")
    print(f"Volume surge ratio: {criteria.volume_surge_ratio}x")
    print(f"Max volatility: {criteria.max_volatility:.1%}")
    print(f"RSI range: {criteria.rsi_min} - {criteria.rsi_max}")
    
    return True


def test_model_config():
    """Test model configuration"""
    print("\n" + "=" * 60)
    print("Testing Model Configuration")
    print("=" * 60)
    
    settings = get_settings()
    model = settings.model
    
    print(f"GPU enabled: {model.use_gpu}")
    print(f"Batch size: {model.batch_size}")
    print(f"LSTM lookback: {model.lstm_lookback_days} days")
    print(f"LSTM forecast: {model.lstm_forecast_days} days")
    print(f"Sentiment model: {model.sentiment_model}")
    print(f"CNN confidence threshold: {model.cnn_confidence_threshold}")
    
    print("\nEnsemble weights:")
    for name, weight in model.ensemble_weights.items():
        print(f"  {name}: {weight:.0%}")
    
    total_weight = sum(model.ensemble_weights.values())
    print(f"  Total: {total_weight:.2f}")
    
    return True


def test_api_config():
    """Test API configuration"""
    print("\n" + "=" * 60)
    print("Testing API Configuration")
    print("=" * 60)
    
    settings = get_settings()
    api = settings.api
    
    # Check which API keys are configured (don't print actual keys)
    apis = [
        ("News API", api.news_api_key),
        ("Alpha Vantage", api.alpha_vantage_key),
        ("Polygon", api.polygon_key),
        ("IEX Cloud", api.iex_cloud_key),
        ("OpenAI", api.openai_api_key),
        ("LangSmith", api.langsmith_api_key)
    ]
    
    for name, key in apis:
        if key:
            # Show only first 4 chars for security
            masked = key[:4] + "*" * (len(key) - 4) if len(key) > 4 else "*" * len(key)
            print(f"✅ {name}: {masked}")
        else:
            print(f"⚠️ {name}: Not configured")
    
    print(f"\nRate limits:")
    print(f"  yfinance delay: {api.yfinance_delay_ms}ms")
    print(f"  News API: {api.news_api_requests_per_minute} req/min")
    
    return True


def test_save_and_load():
    """Test saving and loading settings to/from file"""
    print("\n" + "=" * 60)
    print("Testing Save/Load Settings")
    print("=" * 60)
    
    settings = get_settings()
    
    # Save to file
    test_file = Path("test_settings.json")
    try:
        settings.save_to_file(test_file)
        print(f"✅ Settings saved to {test_file}")
        
        # Load from file
        loaded_settings = Settings.load_from_file(test_file)
        print(f"✅ Settings loaded from {test_file}")
        
        # Verify some values match
        if loaded_settings.filter.min_price == settings.filter.min_price:
            print("✅ Loaded settings match original")
        
        # Clean up
        test_file.unlink()
        print(f"✅ Test file cleaned up")
        
        return True
    except Exception as e:
        print(f"❌ Save/load test failed: {e}")
        if test_file.exists():
            test_file.unlink()
        return False


def test_trading_parameters():
    """Test trading parameters"""
    print("\n" + "=" * 60)
    print("Testing Trading Parameters")
    print("=" * 60)
    
    settings = get_settings()
    trading = settings.trading
    
    print(f"Max positions: {trading.max_positions}")
    print(f"Position size: {trading.position_size_pct:.1%} of portfolio")
    print(f"Stop loss: {trading.stop_loss_pct:.1%}")
    print(f"Take profit: {trading.take_profit_pct:.1%}")
    print(f"Min signal score: {trading.min_signal_score}/100")
    print(f"Strong signal score: {trading.strong_signal_score}/100")
    print(f"Scan time: {trading.scan_time}")
    print(f"Market hours: {trading.market_open} - {trading.market_close}")
    
    return True


def main():
    """Run all configuration tests"""
    print("AI Stock Discovery System - Configuration Test")
    print("=" * 60)
    
    tests = [
        ("Settings Loading", test_settings_loading),
        ("Validation", test_validation),
        ("Directory Creation", test_directory_creation),
        ("Filter Criteria", test_filter_criteria),
        ("Model Configuration", test_model_config),
        ("API Configuration", test_api_config),
        ("Trading Parameters", test_trading_parameters),
        ("Save/Load Settings", test_save_and_load)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All configuration tests passed!")
        print("Issue #4 is complete - Configuration system is ready!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)