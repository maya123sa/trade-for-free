#!/usr/bin/env python3
"""
Run script for Indian Stock Technical Analysis Application
"""

import os
import sys
from app import app
from config import config

def create_app():
    """Create and configure the Flask application"""
    config_name = os.environ.get('FLASK_CONFIG', 'default')
    try:
        app.config.from_object(config[config_name])
    except KeyError:
        print(f"Warning: Configuration '{config_name}' not found, using default")
        app.config.from_object(config['default'])
    
    return app

if __name__ == '__main__':
    # Get configuration
    config_name = os.environ.get('FLASK_CONFIG', 'development')
    
    # Create app
    application = create_app()
    
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # Run the application
    if config_name == 'production':
        # Production mode - use gunicorn
        print("Starting in production mode...")
        print(f"Application will be available at http://0.0.0.0:{port}")
        application.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Development mode
        print("Starting in development mode...")
        print(f"Application will be available at http://localhost:{port}")
        print("Features available:")
        print("- Real-time NSE/BSE stock analysis")
        print("- SuperTrend, Heikin-Ashi, Pivot Points")
        print("- Opening Range Breakout (ORB) scanner")
        print("- Sector heatmap")
        print("- Option chain PCR/OI analysis")
        print("- Telegram/WhatsApp alerts")
        print("- Bilingual support (English/Hindi)")
        print("\nAPI Endpoints:")
        print("- GET /analyze?symbol=RELIANCE - Stock analysis")
        print("- GET /screener - Stock screener")
        print("- GET /momentum?symbol=NIFTY - Momentum analysis")
        print("- GET /alerts?symbol=RELIANCE - Price alerts")
        print("- GET /backtest?symbol=RELIANCE&strategy=rsi_macd - Backtest")
        
        application.run(host='0.0.0.0', port=port, debug=True)