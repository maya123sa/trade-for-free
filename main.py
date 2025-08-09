#!/usr/bin/env python3
"""
Indian Stock Technical Analysis Application
Focused on NSE/BSE markets with Indian trading strategies
"""

import os
import sys
import asyncio
from datetime import datetime, time
import pytz
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from src.data_fetcher import IndianStockDataFetcher
from src.technical_indicators import IndianTechnicalIndicators
from src.trading_strategies import IndianTradingStrategies
from src.visualization import IndianStockVisualizer
from src.alerts import AlertManager
from src.utils import NSEHolidayCalendar, MarketHours

app = Flask(__name__)
CORS(app)

# Initialize components
data_fetcher = IndianStockDataFetcher()
indicators = IndianTechnicalIndicators()
strategies = IndianTradingStrategies()
visualizer = IndianStockVisualizer()
alert_manager = AlertManager()
holiday_calendar = NSEHolidayCalendar()
market_hours = MarketHours()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/api/analyze/<symbol>')
def analyze_stock(symbol):
    """Analyze a specific stock with Indian indicators"""
    try:
        # Fetch data
        data = data_fetcher.get_stock_data(symbol)
        
        # Calculate Indian indicators
        analysis = indicators.calculate_all_indicators(data)
        
        # Generate trading signals
        signals = strategies.generate_signals(data, analysis)
        
        # Check market hours
        market_status = market_hours.get_market_status()
        
        return jsonify({
            'symbol': symbol,
            'market_status': market_status,
            'analysis': analysis,
            'signals': signals,
            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sector-heatmap')
def sector_heatmap():
    """Generate NSE sector heatmap"""
    try:
        heatmap_data = visualizer.generate_sector_heatmap()
        return jsonify(heatmap_data)
    except Exception as e:
        logger.error(f"Error generating sector heatmap: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/orb-scanner')
def orb_scanner():
    """Opening Range Breakout scanner for 9:15-9:45 AM IST"""
    try:
        orb_stocks = strategies.scan_orb_opportunities()
        return jsonify(orb_stocks)
    except Exception as e:
        logger.error(f"Error in ORB scanner: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/fii-dii-activity')
def fii_dii_activity():
    """Get FII/DII activity data"""
    try:
        activity_data = data_fetcher.get_fii_dii_data()
        return jsonify(activity_data)
    except Exception as e:
        logger.error(f"Error fetching FII/DII data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/option-chain/<symbol>')
def option_chain(symbol):
    """Get option chain data with PCR/OI analysis"""
    try:
        option_data = data_fetcher.get_option_chain_data(symbol)
        analysis = indicators.analyze_option_chain(option_data)
        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error fetching option chain for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk-deals')
def bulk_deals():
    """Get bulk and block deals data"""
    try:
        deals_data = data_fetcher.get_bulk_block_deals()
        return jsonify(deals_data)
    except Exception as e:
        logger.error(f"Error fetching bulk deals: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check if market is open
    if market_hours.is_market_open():
        logger.info("Market is open - Starting real-time monitoring")
    else:
        logger.info("Market is closed - Running in analysis mode")
    
    app.run(host='0.0.0.0', port=5000, debug=True)