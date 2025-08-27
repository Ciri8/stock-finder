"""
Configuration management system for AI Stock Discovery System.
Handles all settings, API keys, and trading parameters with validation.

right now breakout_filter.py is working just fine with hardcoded settings but later i will switch to this.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import time
from pathlib import Path
from enum import Enum
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MarketRegime(Enum):
    """Market regime types for adaptive strategies"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class DataSource(Enum):
    """Available data sources"""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IEX_CLOUD = "iex_cloud"


@dataclass
class APIConfig:
    """API configuration and keys"""
    # Required - Free tier available
    news_api_key: Optional[str] = field(default_factory=lambda: os.getenv("NEWS_API_KEY"))
    
    # Ollama configuration (FREE - runs locally)
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama2"))  # or mistral, phi, etc.
    
    # Optional backup data sources (only if yfinance fails)
    alpha_vantage_key: Optional[str] = field(default_factory=lambda: os.getenv("ALPHA_VANTAGE_KEY"))
    polygon_key: Optional[str] = field(default_factory=lambda: os.getenv("POLYGON_KEY"))
    iex_cloud_key: Optional[str] = field(default_factory=lambda: os.getenv("IEX_CLOUD_KEY"))
    
    # Rate limiting
    yfinance_delay_ms: int = 100  # Delay between yfinance requests
    news_api_requests_per_minute: int = 100
    
    def validate(self) -> List[str]:
        """Validate API configuration"""
        errors = []
        if self.news_api_key and len(self.news_api_key) < 10:
            errors.append("NEWS_API_KEY appears to be invalid (too short)")
        return errors


@dataclass
class DataConfig:
    """Data fetching and storage configuration"""
    # Data sources
    primary_source: DataSource = DataSource.YFINANCE
    fallback_source: Optional[DataSource] = None
    
    # Time ranges
    historical_days: int = 60  # Days of historical data to fetch
    intraday_interval: str = "5m"  # For intraday data
    daily_update_time: time = time(18, 0)  # 6 PM EST
    
    # Cache settings
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    
    # Data paths
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw"))
    processed_data_dir: Path = field(default_factory=lambda: Path("data/processed"))
    models_dir: Path = field(default_factory=lambda: Path("data/models"))
    reports_dir: Path = field(default_factory=lambda: Path("data/reports"))
    logs_dir: Path = field(default_factory=lambda: Path("data/logs"))
    
    def validate(self) -> List[str]:
        """Validate data configuration"""
        errors = []
        if self.historical_days < 20:
            errors.append("historical_days should be at least 20 for meaningful analysis")
        if self.historical_days > 365:
            errors.append("historical_days > 365 may cause performance issues")
        return errors


@dataclass
class FilterCriteria:
    """Stock filtering criteria"""
    # Price filters
    min_price: float = 20.0  # Minimum stock price
    max_price: float = 10000.0  # Maximum stock price
    
    # Volume filters
    min_volume: int = 1_000_000  # Minimum daily volume
    min_avg_volume: int = 500_000  # Minimum 20-day average volume
    volume_surge_ratio: float = 1.5  # Current volume vs average ratio
    
    # Performance filters
    min_weekly_change: float = 0.04  # 4% minimum weekly gain
    max_weekly_change: float = 0.10  # 10% maximum (avoid overextended)
    min_daily_change: float = 0.01  # 1% minimum daily change
    
    # Volatility filters
    max_volatility: float = 0.15  # 15% maximum volatility
    min_volatility: float = 0.02  # 2% minimum (avoid dead stocks)
    
    # Quality filters
    min_market_cap: Optional[float] = 1_000_000_000  # $1B minimum
    max_pe_ratio: Optional[float] = 50.0  # Maximum P/E ratio
    min_relative_volume: float = 1.2  # Volume relative to average
    
    # Technical filters
    above_sma_20: bool = True  # Price above 20-day SMA
    above_sma_50: bool = False  # Price above 50-day SMA
    rsi_min: float = 30.0  # Minimum RSI (avoid oversold)
    rsi_max: float = 70.0  # Maximum RSI (avoid overbought)
    
    def validate(self) -> List[str]:
        """Validate filter criteria"""
        errors = []
        if self.min_price >= self.max_price:
            errors.append("min_price must be less than max_price")
        if self.min_weekly_change >= self.max_weekly_change:
            errors.append("min_weekly_change must be less than max_weekly_change")
        if self.rsi_min >= self.rsi_max:
            errors.append("rsi_min must be less than rsi_max")
        if self.volume_surge_ratio < 1.0:
            errors.append("volume_surge_ratio must be >= 1.0")
        return errors


@dataclass
class ModelConfig:
    """AI model configuration"""
    # Model paths
    cnn_model_path: Optional[Path] = field(default_factory=lambda: Path("data/models/cnn_pattern.pth"))
    lstm_model_path: Optional[Path] = field(default_factory=lambda: Path("data/models/lstm_price.pth"))
    xgboost_model_path: Optional[Path] = field(default_factory=lambda: Path("data/models/xgb_risk.pkl"))
    
    # Model parameters
    cnn_confidence_threshold: float = 0.7  # Minimum confidence for pattern detection
    lstm_lookback_days: int = 60  # Days of history for LSTM
    lstm_forecast_days: int = 5  # Days to predict ahead
    
    # Sentiment analysis
    sentiment_model: str = "ProsusAI/finbert"  # HuggingFace model
    sentiment_lookback_hours: int = 48  # Hours of news to analyze
    sentiment_min_articles: int = 3  # Minimum articles for valid sentiment
    
    # Risk model
    risk_features: List[str] = field(default_factory=lambda: [
        "volatility", "beta", "sharpe_ratio", "max_drawdown",
        "volume_ratio", "price_momentum", "rsi", "macd"
    ])
    
    # Ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "technical": 0.3,  # CNN patterns
        "quantitative": 0.25,  # LSTM predictions
        "sentiment": 0.2,  # News sentiment
        "risk": 0.15,  # Risk score
        "momentum": 0.1  # Price/volume momentum
    })
    
    # GPU settings
    use_gpu: bool = True
    gpu_device: int = 0  # CUDA device ID
    batch_size: int = 32
    
    def validate(self) -> List[str]:
        """Validate model configuration"""
        errors = []
        weights_sum = sum(self.ensemble_weights.values())
        if abs(weights_sum - 1.0) > 0.01:
            errors.append(f"ensemble_weights must sum to 1.0, got {weights_sum}")
        if self.lstm_lookback_days < 20:
            errors.append("lstm_lookback_days should be at least 20")
        if self.cnn_confidence_threshold < 0 or self.cnn_confidence_threshold > 1:
            errors.append("cnn_confidence_threshold must be between 0 and 1")
        return errors


@dataclass
class TradingParameters:
    """Trading and analysis parameters"""
    # Position sizing
    max_positions: int = 20  # Maximum concurrent positions
    position_size_pct: float = 0.05  # 5% of portfolio per position
    
    # Risk management
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    trailing_stop_pct: float = 0.03  # 3% trailing stop
    
    # Signal thresholds
    min_signal_score: float = 70.0  # Minimum score (0-100) to consider
    strong_signal_score: float = 85.0  # Score for strong buy signal
    
    # Market hours (EST)
    market_open: time = time(9, 30)
    market_close: time = time(16, 0)
    scan_time: time = time(18, 0)  # When to run daily scan
    
    # Backtesting
    backtest_start_date: str = "2019-01-01"
    backtest_end_date: str = "2024-01-01"
    backtest_initial_capital: float = 100000.0
    backtest_commission: float = 0.001  # 0.1% commission
    
    def validate(self) -> List[str]:
        """Validate trading parameters"""
        errors = []
        if self.position_size_pct > 0.1:
            errors.append("position_size_pct > 10% is risky")
        if self.stop_loss_pct > 0.1:
            errors.append("stop_loss_pct > 10% is too high")
        if self.min_signal_score < 0 or self.min_signal_score > 100:
            errors.append("min_signal_score must be between 0 and 100")
        return errors


@dataclass
class NotificationConfig:
    """Notification and alerting configuration"""
    # Email settings
    email_enabled: bool = False
    smtp_server: Optional[str] = field(default_factory=lambda: os.getenv("SMTP_SERVER"))
    smtp_port: int = 587
    email_from: Optional[str] = field(default_factory=lambda: os.getenv("EMAIL_FROM"))
    email_to: List[str] = field(default_factory=list)
    email_password: Optional[str] = field(default_factory=lambda: os.getenv("EMAIL_PASSWORD"))
    
    # Slack settings
    slack_enabled: bool = False
    slack_webhook_url: Optional[str] = field(default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL"))
    slack_channel: str = "#trading-signals"
    
    # Notification thresholds
    alert_on_strong_signal: bool = True  # Alert when score > strong_signal_score
    alert_on_high_volume: bool = True  # Alert on unusual volume
    alert_on_pattern: bool = True  # Alert on specific patterns
    
    # Report settings
    generate_daily_report: bool = True
    report_format: str = "pdf"  # pdf, html, csv
    include_charts: bool = True
    
    def validate(self) -> List[str]:
        """Validate notification configuration"""
        errors = []
        if self.email_enabled and not self.smtp_server:
            errors.append("SMTP_SERVER required when email_enabled=True")
        if self.slack_enabled and not self.slack_webhook_url:
            errors.append("SLACK_WEBHOOK_URL required when slack_enabled=True")
        return errors


@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Environment
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_to_file: bool = True
    log_rotation: str = "daily"  # daily, weekly, size
    log_retention_days: int = 30
    
    # Performance
    max_workers: int = 4  # Parallel processing workers
    chunk_size: int = 50  # Stocks to process per chunk
    enable_profiling: bool = False
    
    # Database (future enhancement)
    use_database: bool = False
    database_url: Optional[str] = field(default_factory=lambda: os.getenv("DATABASE_URL"))
    
    def validate(self) -> List[str]:
        """Validate system configuration"""
        errors = []
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")
        if self.max_workers < 1 or self.max_workers > 16:
            errors.append("max_workers should be between 1 and 16")
        return errors


@dataclass
class Settings:
    """Main settings container combining all configurations"""
    api: APIConfig = field(default_factory=APIConfig)
    data: DataConfig = field(default_factory=DataConfig)
    filter: FilterCriteria = field(default_factory=FilterCriteria)
    model: ModelConfig = field(default_factory=ModelConfig)
    trading: TradingParameters = field(default_factory=TradingParameters)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def validate(self) -> Dict[str, List[str]]:
        """Validate all configurations"""
        all_errors = {}
        
        # Validate each section
        for section_name, section in [
            ("api", self.api),
            ("data", self.data),
            ("filter", self.filter),
            ("model", self.model),
            ("trading", self.trading),
            ("notifications", self.notifications),
            ("system", self.system)
        ]:
            errors = section.validate()
            if errors:
                all_errors[section_name] = errors
        
        # Cross-section validation
        if self.model.lstm_lookback_days > self.data.historical_days:
            all_errors.setdefault("cross_validation", []).append(
                f"model.lstm_lookback_days ({self.model.lstm_lookback_days}) > "
                f"data.historical_days ({self.data.historical_days})"
            )
        
        return all_errors
    
    def create_directories(self):
        """Create all required directories"""
        directories = [
            self.data.cache_dir,
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.models_dir,
            self.data.reports_dir,
            self.data.logs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "api": {k: v for k, v in self.api.__dict__.items() if not k.startswith('_')},
            "data": {k: str(v) if isinstance(v, Path) else v 
                    for k, v in self.data.__dict__.items()},
            "filter": self.filter.__dict__,
            "model": {k: str(v) if isinstance(v, Path) else v 
                     for k, v in self.model.__dict__.items()},
            "trading": self.trading.__dict__,
            "notifications": self.notifications.__dict__,
            "system": self.system.__dict__
        }
    
    def save_to_file(self, filepath: Path):
        """Save settings to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'Settings':
        """Load settings from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert back to proper types
        settings = cls()
        
        # This is simplified - in production you'd want proper deserialization
        for section_name, section_data in data.items():
            if hasattr(settings, section_name):
                section = getattr(settings, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
        
        return settings
    
    def print_summary(self):
        """Print a summary of key settings"""
        print("=" * 60)
        print("AI Stock Discovery System - Configuration Summary")
        print("=" * 60)
        print(f"Environment: {self.system.environment}")
        print(f"Debug Mode: {self.system.debug_mode}")
        print(f"Primary Data Source: {self.data.primary_source.value}")
        print(f"Historical Days: {self.data.historical_days}")
        print(f"GPU Enabled: {self.model.use_gpu}")
        print(f"Min Signal Score: {self.trading.min_signal_score}")
        print(f"Email Notifications: {self.notifications.email_enabled}")
        print(f"Slack Notifications: {self.notifications.slack_enabled}")
        print("=" * 60)


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create singleton settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.create_directories()
        
        # Validate on first load
        errors = _settings.validate()
        if errors:
            print("Warning: Configuration validation errors found:")
            for section, section_errors in errors.items():
                for error in section_errors:
                    print(f"  [{section}] {error}")
    
    return _settings


def reload_settings():
    """Force reload of settings"""
    global _settings
    _settings = None
    return get_settings()


# Convenience exports
settings = get_settings()
api_config = settings.api
data_config = settings.data
filter_criteria = settings.filter
model_config = settings.model
trading_params = settings.trading
notification_config = settings.notifications
system_config = settings.system