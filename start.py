#!/usr/bin/env python3
"""
Simple start script for the Indian Stock Analysis application
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    
    if __name__ == '__main__':
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
        
        print("🚀 Starting Indian Stock Technical Analysis Application")
        print(f"📊 Server running on http://localhost:{port}")
        print("🔗 API endpoints available:")
        print("   - /analyze?symbol=RELIANCE")
        print("   - /screener")
        print("   - /momentum?symbol=NIFTY")
        print("   - /alerts?symbol=RELIANCE")
        print("   - /backtest?symbol=RELIANCE&strategy=rsi_macd")
        print("\n✨ Features:")
        print("   - Real-time NSE/BSE stock analysis")
        print("   - Technical indicators (RSI, MACD, Bollinger Bands)")
        print("   - Stock screening and momentum analysis")
        print("   - Price alerts and backtesting")
        print("   - Interactive charts and visualizations")
        
        app.run(host='0.0.0.0', port=port, debug=debug)
        
except ImportError as e:
    print(f"❌ Error importing app: {e}")
    print("📦 Please install required dependencies:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting application: {e}")
    sys.exit(1)