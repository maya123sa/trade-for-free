"""
Data fetcher for Indian stock markets (NSE/BSE)
Handles real-time data, FII/DII activity, and option chains
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
from tradingview_ta import TA_Handler, Interval, Exchange
import yfinance as yf
import warnings
import logging

# Suppress yfinance warnings
warnings.filterwarnings("ignore", message=".*yfinance.*")

logger = logging.getLogger(__name__)

class IndianStockDataFetcher:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.nse_symbols = {
            'NIFTY': '^NSEI',
            'SENSEX': '^BSESN',
            'BANKNIFTY': '^NSEBANK',
            'HDFCBANK': 'HDFCBANK.NS',
            'INFY': 'INFY.NS',
            'TATAMOTORS': 'TATAMOTORS.NS',
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'SBIN': 'SBIN.NS',
            'ITC': 'ITC.NS',
            'HINDUNILVR': 'HINDUNILVR.NS'
        }
        
        self.sector_stocks = {
            'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
            'IT': ['INFY', 'TCS', 'WIPRO', 'HCLTECH', 'TECHM'],
            'Auto': ['TATAMOTORS', 'M&M', 'MARUTI', 'BAJAJ-AUTO', 'HEROMOTOCO'],
            'Pharma': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'DIVISLAB', 'BIOCON'],
            'FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
            'Energy': ['RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'COALINDIA'],
            'Metals': ['TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VEDL', 'NMDC']
        }

    def get_stock_data(self, symbol, interval='1d', period='1y'):
        """Fetch stock data with NSE/BSE format handling"""
        try:
            # Convert to Yahoo Finance format
            if symbol in self.nse_symbols:
                yf_symbol = self.nse_symbols[symbol]
            elif not symbol.endswith(('.NS', '.BO')):
                yf_symbol = f"{symbol}.NS"  # Default to NSE
            else:
                yf_symbol = symbol
            
            # Fetch data
            stock = yf.Ticker(yf_symbol)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = stock.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol: {symbol}")
            
            # Add symbol info
            data.attrs['symbol'] = symbol
            data.attrs['yf_symbol'] = yf_symbol
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise

    def get_tradingview_analysis(self, symbol, interval=Interval.INTERVAL_1_DAY):
        """Get TradingView technical analysis for Indian stocks"""
        try:
            # Clean symbol for TradingView
            tv_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            handler = TA_Handler(
                symbol=tv_symbol,
                screener="india",
                exchange="NSE",
                interval=interval
            )
            
            analysis = handler.get_analysis()
            
            return {
                'recommendation': analysis.summary['RECOMMENDATION'],
                'buy_signals': analysis.summary['BUY'],
                'sell_signals': analysis.summary['SELL'],
                'neutral_signals': analysis.summary['NEUTRAL'],
                'indicators': analysis.indicators,
                'oscillators': analysis.oscillators,
                'moving_averages': analysis.moving_averages
            }
            
        except Exception as e:
            logger.error(f"Error getting TradingView analysis for {symbol}: {str(e)}")
            return None

    def get_nifty_50_stocks(self):
        """Get current Nifty 50 constituent stocks"""
        nifty_50 = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK',
            'HDFC', 'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'AXISBANK',
            'ASIANPAINT', 'MARUTI', 'NESTLEIND', 'HCLTECH', 'BAJFINANCE',
            'M&M', 'SUNPHARMA', 'TITAN', 'ULTRACEMCO', 'WIPRO', 'NTPC',
            'JSWSTEEL', 'POWERGRID', 'TATASTEEL', 'TECHM', 'HINDALCO',
            'COALINDIA', 'INDUSINDBK', 'GRASIM', 'CIPLA', 'DRREDDY',
            'EICHERMOT', 'UPL', 'BAJAJ-AUTO', 'ONGC', 'TATAMOTORS',
            'SBILIFE', 'HDFCLIFE', 'BRITANNIA', 'ADANIPORTS', 'HEROMOTOCO',
            'DIVISLAB', 'SHREECEM', 'BAJAJFINSV', 'BPCL', 'IOC', 'TATACONSUM'
        ]
        return nifty_50

    def get_fii_dii_data(self):
        """Fetch FII/DII activity data (mock implementation)"""
        try:
            # In real implementation, this would fetch from NSE API or data provider
            # For now, returning mock data structure
            current_date = datetime.now(self.ist).strftime('%Y-%m-%d')
            
            mock_data = {
                'date': current_date,
                'fii': {
                    'equity_buy': 2500.50,
                    'equity_sell': 2300.25,
                    'equity_net': 200.25,
                    'debt_buy': 150.75,
                    'debt_sell': 175.50,
                    'debt_net': -24.75
                },
                'dii': {
                    'equity_buy': 1800.30,
                    'equity_sell': 1650.80,
                    'equity_net': 149.50,
                    'debt_buy': 300.25,
                    'debt_sell': 280.15,
                    'debt_net': 20.10
                }
            }
            
            return mock_data
            
        except Exception as e:
            logger.error(f"Error fetching FII/DII data: {str(e)}")
            return None

    def get_option_chain_data(self, symbol):
        """Fetch option chain data for analysis"""
        try:
            # Mock option chain data structure
            # In real implementation, would fetch from NSE option chain API
            
            mock_option_data = {
                'symbol': symbol,
                'expiry_date': '2024-01-25',
                'spot_price': 18500.0,
                'calls': [
                    {'strike': 18000, 'oi': 50000, 'volume': 1200, 'ltp': 520.5},
                    {'strike': 18500, 'oi': 75000, 'volume': 2500, 'ltp': 280.3},
                    {'strike': 19000, 'oi': 60000, 'volume': 1800, 'ltp': 120.8}
                ],
                'puts': [
                    {'strike': 18000, 'oi': 45000, 'volume': 1000, 'ltp': 95.2},
                    {'strike': 18500, 'oi': 80000, 'volume': 2200, 'ltp': 250.7},
                    {'strike': 19000, 'oi': 55000, 'volume': 1500, 'ltp': 480.5}
                ]
            }
            
            return mock_option_data
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {str(e)}")
            return None

    def get_bulk_block_deals(self):
        """Fetch bulk and block deals data"""
        try:
            # Mock bulk deals data
            # In real implementation, would scrape from NSE website
            
            current_date = datetime.now(self.ist).strftime('%Y-%m-%d')
            
            mock_deals = {
                'date': current_date,
                'bulk_deals': [
                    {
                        'symbol': 'RELIANCE',
                        'client_name': 'ABC MUTUAL FUND',
                        'buy_sell': 'BUY',
                        'quantity': 100000,
                        'price': 2450.50
                    }
                ],
                'block_deals': [
                    {
                        'symbol': 'TCS',
                        'client_name': 'XYZ INVESTMENTS',
                        'buy_sell': 'SELL',
                        'quantity': 500000,
                        'price': 3680.25
                    }
                ]
            }
            
            return mock_deals
            
        except Exception as e:
            logger.error(f"Error fetching bulk deals: {str(e)}")
            return None

    def get_sector_performance(self):
        """Get sector-wise performance data"""
        try:
            sector_data = {}
            
            for sector, stocks in self.sector_stocks.items():
                sector_performance = []
                
                for stock in stocks[:3]:  # Limit to top 3 stocks per sector
                    try:
                        data = self.get_stock_data(stock, period='5d')
                        if not data.empty:
                            change_pct = ((data['Close'][-1] - data['Close'][-2]) / data['Close'][-2]) * 100
                            sector_performance.append({
                                'symbol': stock,
                                'price': data['Close'][-1],
                                'change_pct': round(change_pct, 2)
                            })
                    except:
                        continue
                
                if sector_performance:
                    avg_change = sum([s['change_pct'] for s in sector_performance]) / len(sector_performance)
                    sector_data[sector] = {
                        'avg_change': round(avg_change, 2),
                        'stocks': sector_performance
                    }
            
            return sector_data
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {str(e)}")
            return {}