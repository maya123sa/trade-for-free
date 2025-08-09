# Indian Stock Technical Analysis Application

A comprehensive Python application for technical analysis of Indian stocks (NSE/BSE) with focus on Indian trading strategies and indicators.

## 🚀 Features

### Core Analysis
- **Real-time Data**: NSE/BSE markets with `screener="india"`
- **Indian Indicators**: SuperTrend (7,3), Heikin-Ashi, Pivot Points, EMA(20)/SMA(50,200)
- **Volume Profile**: POC/VAH/VAL analysis
- **TradingView Integration**: Real-time technical analysis

### Trading Strategies
- **Opening Range Breakout (ORB)**: 9:15-9:45 AM IST detection
- **SuperTrend + EMA Crossover**: Combined signal generation
- **Nifty Correlation**: Index correlation scoring
- **Multi-timeframe Analysis**: 1m to 1W intervals

### Market Coverage
- **Indices**: NIFTY 50, SENSEX, BANK NIFTY
- **Top Sectors**: Banking (HDFCBANK), IT (INFY), Auto (TATAMOTORS)
- **50+ Stocks**: Complete Nifty 50 coverage
- **F&O Stocks**: Futures & Options analysis

### Visualization
- **Market Hours**: Pre-open, Open, Close annotations (IST)
- **Sector Heatmap**: NSE sector performance
- **Option Chain**: PCR/OI visualization
- **Interactive Charts**: Candlestick with indicators

### Alerts & Integration
- **Telegram Alerts**: Real-time trading signals
- **WhatsApp Support**: Message notifications
- **Bilingual Reports**: English/Hindi support
- **Broker Integration**: Zerodha/Upstox hooks

### Advanced Features
- **FII/DII Activity**: Foreign/Domestic institutional data
- **Bulk Deal Detection**: Large block transactions
- **NSE Holiday Calendar**: Trading day management
- **Risk Management**: Position sizing and stop-loss

## 📦 Installation

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Required Packages
```bash
pip install Flask Flask-CORS yfinance pandas numpy matplotlib seaborn
pip install tradingview-ta TA-Lib requests beautifulsoup4 pytz scikit-learn
```

### TA-Lib Installation
```bash
# Ubuntu/Debian
sudo apt-get install libta-lib-dev

# macOS
brew install ta-lib

# Windows
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```

## 🚀 Quick Start

### 1. Basic Usage
```bash
python run.py
```

### 2. Access Dashboard
```
http://localhost:5000
```

### 3. Environment Variables
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token"
export TELEGRAM_CHAT_ID="your_chat_id"
export ZERODHA_API_KEY="your_api_key"
export UPSTOX_API_KEY="your_api_key"
```

## 📊 API Endpoints

### Stock Analysis
```bash
GET /api/analyze/RELIANCE
GET /api/analyze/NIFTY
GET /api/analyze/BANKNIFTY
```

### ORB Scanner
```bash
GET /api/orb-scanner
```

### Sector Analysis
```bash
GET /api/sector-heatmap
```

### Option Chain
```bash
GET /api/option-chain/NIFTY
GET /api/option-chain/BANKNIFTY
```

### Market Data
```bash
GET /api/fii-dii-activity
GET /api/bulk-deals
```

## 🔧 Configuration

### Market Hours (IST)
- **Pre-open**: 9:00-9:15 AM
- **Trading**: 9:15 AM-3:30 PM
- **Closing**: 3:40-4:00 PM

### Technical Indicators
- **SuperTrend**: Period=7, Multiplier=3.0
- **EMA**: 20 period
- **SMA**: 50, 200 periods
- **Volume Profile**: 20 bins
- **ORB**: 30-minute range (9:15-9:45 AM)

### Supported Symbols
```python
# Indices
NIFTY, SENSEX, BANKNIFTY

# Top Stocks
RELIANCE, TCS, HDFCBANK, INFY, HINDUNILVR
ICICIBANK, SBIN, ITC, KOTAKBANK, LT

# Sectors
Banking, IT, Auto, Pharma, FMCG, Energy, Metals
```

## 📱 Alert Configuration

### Telegram Setup
1. Create bot with @BotFather
2. Get bot token
3. Get chat ID
4. Set environment variables

### WhatsApp Setup
1. Configure Twilio account
2. Set webhook URL
3. Enable WhatsApp sandbox

## 🔌 Broker Integration

### Zerodha KiteConnect
```python
from src.utils import BrokerageIntegration

broker = BrokerageIntegration()
broker.configure_zerodha(api_key, access_token)
```

### Upstox API
```python
broker.configure_upstox(api_key, access_token)
```

## 📈 Trading Strategies

### Opening Range Breakout (ORB)
```python
# Detect ORB setup
orb_data = strategies.detect_orb_setup(data)

# Key levels
orb_high = orb_data['orb_high']
orb_low = orb_data['orb_low']
targets = [orb_data['target1'], orb_data['target2']]
```

### SuperTrend + EMA
```python
# Combined signal
signal = strategies.supertrend_ema_crossover(data, indicators)

# Signal types: STRONG_BUY, BUY, SELL, STRONG_SELL
```

### Volume Analysis
```python
# Volume spike detection
volume_signal = strategies._analyze_volume(data)

# High volume breakouts
```

## 🌐 Bilingual Support

### Hindi Translations
```python
translations = {
    'hi': {
        'buy_signal': 'खरीदारी संकेत',
        'sell_signal': 'बिक्री संकेत',
        'current_price': 'वर्तमान मूल्य'
    }
}
```

### Usage
```python
alert_manager.send_trading_alert(symbol, signals, language='hi')
```

## 📊 Data Sources

### Primary Sources
- **Yahoo Finance**: Historical and real-time data
- **TradingView**: Technical analysis
- **NSE Website**: Official market data

### Backup Sources
- **Alpha Vantage**: Alternative data provider
- **Quandl**: Economic data
- **RBI**: Interest rates and policy data

## 🛡️ Risk Management

### Position Sizing
```python
# Maximum 2% risk per trade
position_size = calculate_position_size(
    account_value=100000,
    risk_percent=2.0,
    stop_loss_distance=50
)
```

### Stop Loss Calculation
```python
# SuperTrend-based stops
stop_loss = supertrend_value
target = entry_price + (2 * (entry_price - stop_loss))
```

## 📅 Market Calendar

### NSE Holidays 2024
- Republic Day: Jan 26
- Holi: Mar 8, 25
- Good Friday: Mar 29
- Independence Day: Aug 15
- Gandhi Jayanti: Oct 2
- Diwali: Nov 1
- Christmas: Dec 25

### Trading Sessions
```python
# Regular session
market_open = time(9, 15)
market_close = time(15, 30)

# Pre-open session
pre_open = time(9, 0)
pre_open_end = time(9, 15)
```

## 🔍 Monitoring & Logging

### Log Levels
```python
import logging
logging.basicConfig(level=logging.INFO)

# Log files
logs/
├── app.log
├── trading.log
├── alerts.log
└── errors.log
```

### Performance Metrics
- API response times
- Data fetch success rates
- Alert delivery status
- Trading signal accuracy

## 🚀 Deployment

### Local Development
```bash
export FLASK_CONFIG=development
python run.py
```

### Production
```bash
export FLASK_CONFIG=production
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

### Docker
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

### Code Style
- PEP 8 compliance
- Type hints
- Docstrings
- Unit tests

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

### Documentation
- API documentation: `/docs`
- Trading strategies: `/docs/strategies`
- Configuration guide: `/docs/config`

### Community
- GitHub Issues
- Telegram Group
- Discord Server

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It is not financial advice. Trading in stocks involves risk and you should consult with a qualified financial advisor before making any investment decisions.

## 🔄 Updates

### Version 1.0.0
- Initial release
- Basic technical analysis
- ORB scanner
- Telegram alerts

### Roadmap
- [ ] Machine learning predictions
- [ ] Backtesting engine
- [ ] Portfolio management
- [ ] Mobile app
- [ ] Advanced charting
- [ ] Social trading features

---

**Made with ❤️ for Indian Stock Markets**