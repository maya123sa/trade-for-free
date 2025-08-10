"""
Configuration settings for Indian Stock Analysis Application
"""

import os
from datetime import time
import pytz

class Config:
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'indian-stock-analysis-secret-key'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Timezone
    TIMEZONE = pytz.timezone('Asia/Kolkata')
    
    # Market hours (IST)
    MARKET_PRE_OPEN = time(9, 0)    # 9:00 AM
    MARKET_OPEN = time(9, 15)       # 9:15 AM
    MARKET_CLOSE = time(15, 30)     # 3:30 PM
    
    # Alert settings
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
    WHATSAPP_ENABLED = os.environ.get('WHATSAPP_ENABLED', 'False').lower() == 'true'
    
    # Data refresh intervals (seconds)
    MARKET_STATUS_REFRESH = 30
    STOCK_DATA_REFRESH = 60
    ORB_SCAN_REFRESH = 300  # 5 minutes
    SECTOR_DATA_REFRESH = 600  # 10 minutes
    
    # Technical indicator settings
    SUPERTREND_PERIOD = 7
    SUPERTREND_MULTIPLIER = 3.0
    EMA_PERIOD = 20
    SMA_PERIODS = [50, 200]
    RSI_PERIOD = 14
    VOLUME_PROFILE_BINS = 20
    
    # ORB settings
    ORB_DURATION_MINUTES = 30  # 9:15-9:45 AM
    ORB_MIN_RANGE_PERCENT = 0.5  # Minimum 0.5% range
    ORB_MAX_STOCKS_SCAN = 50
    
    # Risk management
    DEFAULT_RISK_REWARD_RATIO = 2.0
    MAX_POSITION_SIZE_PERCENT = 2.0  # 2% of portfolio
    
    # Supported languages
    SUPPORTED_LANGUAGES = ['en', 'hi']
    DEFAULT_LANGUAGE = 'en'

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    
class TestingConfig(Config):
    TESTING = True
    DEBUG = True

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}