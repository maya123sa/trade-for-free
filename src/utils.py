"""
Utility functions for Indian stock analysis
NSE holiday calendar, market hours, symbol formatting
"""

import pandas as pd
from datetime import datetime, time, date
import pytz
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class NSEHolidayCalendar:
    def __init__(self):
        # NSE holidays for 2024 (update annually)
        self.holidays_2024 = [
            date(2024, 1, 26),  # Republic Day
            date(2024, 3, 8),   # Holi
            date(2024, 3, 25),  # Holi (Second day)
            date(2024, 3, 29),  # Good Friday
            date(2024, 4, 11),  # Id-Ul-Fitr
            date(2024, 4, 14),  # Dr. Baba Saheb Ambedkar Jayanti
            date(2024, 4, 17),  # Ram Navami
            date(2024, 5, 1),   # Maharashtra Day
            date(2024, 6, 17),  # Bakri Id
            date(2024, 8, 15),  # Independence Day
            date(2024, 8, 26),  # Janmashtami
            date(2024, 10, 2),  # Gandhi Jayanti
            date(2024, 11, 1),  # Diwali Laxmi Pujan
            date(2024, 11, 15), # Guru Nanak Jayanti
            date(2024, 12, 25), # Christmas
        ]
        
        # Add weekends
        self.ist = pytz.timezone('Asia/Kolkata')

    def is_trading_day(self, check_date: date = None) -> bool:
        """Check if given date is a trading day"""
        try:
            if check_date is None:
                check_date = datetime.now(self.ist).date()
            
            # Check if weekend
            if check_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Check if holiday
            if check_date in self.holidays_2024:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking trading day: {str(e)}")
            return False

    def get_next_trading_day(self, from_date: date = None) -> date:
        """Get next trading day"""
        try:
            if from_date is None:
                from_date = datetime.now(self.ist).date()
            
            next_date = from_date
            while not self.is_trading_day(next_date):
                next_date = pd.Timestamp(next_date) + pd.Timedelta(days=1)
                next_date = next_date.date()
            
            return next_date
            
        except Exception as e:
            logger.error(f"Error getting next trading day: {str(e)}")
            return from_date

    def get_trading_days_in_month(self, year: int, month: int) -> List[date]:
        """Get all trading days in a given month"""
        try:
            # Get all days in month
            start_date = date(year, month, 1)
            if month == 12:
                end_date = date(year + 1, 1, 1)
            else:
                end_date = date(year, month + 1, 1)
            
            trading_days = []
            current_date = start_date
            
            while current_date < end_date:
                if self.is_trading_day(current_date):
                    trading_days.append(current_date)
                current_date = pd.Timestamp(current_date) + pd.Timedelta(days=1)
                current_date = current_date.date()
            
            return trading_days
            
        except Exception as e:
            logger.error(f"Error getting trading days for {year}-{month}: {str(e)}")
            return []

class MarketHours:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Market timings
        self.pre_open_start = time(9, 0)    # 9:00 AM
        self.pre_open_end = time(9, 15)     # 9:15 AM
        self.market_open = time(9, 15)      # 9:15 AM
        self.market_close = time(15, 30)    # 3:30 PM
        
        # Special sessions
        self.closing_session_start = time(15, 40)  # 3:40 PM
        self.closing_session_end = time(16, 0)     # 4:00 PM

    def get_market_status(self) -> Dict[str, Any]:
        """Get current market status"""
        try:
            now = datetime.now(self.ist)
            current_time = now.time()
            current_date = now.date()
            
            # Check if trading day
            holiday_calendar = NSEHolidayCalendar()
            is_trading_day = holiday_calendar.is_trading_day(current_date)
            
            if not is_trading_day:
                return {
                    'status': 'CLOSED',
                    'reason': 'Holiday/Weekend',
                    'next_open': holiday_calendar.get_next_trading_day(current_date),
                    'current_time': now.strftime('%H:%M:%S IST')
                }
            
            # Determine market status
            if self.pre_open_start <= current_time < self.pre_open_end:
                status = 'PRE_OPEN'
                reason = 'Pre-market session'
            elif self.market_open <= current_time < self.market_close:
                status = 'OPEN'
                reason = 'Regular trading session'
            elif self.closing_session_start <= current_time < self.closing_session_end:
                status = 'CLOSING_SESSION'
                reason = 'Closing session'
            else:
                status = 'CLOSED'
                reason = 'After market hours'
            
            return {
                'status': status,
                'reason': reason,
                'current_time': now.strftime('%H:%M:%S IST'),
                'market_open': self.market_open.strftime('%H:%M'),
                'market_close': self.market_close.strftime('%H:%M'),
                'is_trading_day': is_trading_day
            }
            
        except Exception as e:
            logger.error(f"Error getting market status: {str(e)}")
            return {'status': 'ERROR', 'reason': str(e)}

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        try:
            status = self.get_market_status()
            return status['status'] in ['OPEN', 'PRE_OPEN']
        except:
            return False

    def time_to_market_open(self) -> Optional[int]:
        """Get minutes until market opens"""
        try:
            now = datetime.now(self.ist)
            current_time = now.time()
            
            if current_time < self.market_open:
                # Same day opening
                market_open_today = datetime.combine(now.date(), self.market_open)
                market_open_today = self.ist.localize(market_open_today)
                diff = market_open_today - now
                return int(diff.total_seconds() / 60)
            else:
                # Next trading day
                holiday_calendar = NSEHolidayCalendar()
                next_trading_day = holiday_calendar.get_next_trading_day(now.date() + pd.Timedelta(days=1))
                next_open = datetime.combine(next_trading_day, self.market_open)
                next_open = self.ist.localize(next_open)
                diff = next_open - now
                return int(diff.total_seconds() / 60)
                
        except Exception as e:
            logger.error(f"Error calculating time to market open: {str(e)}")
            return None

    def time_to_market_close(self) -> Optional[int]:
        """Get minutes until market closes"""
        try:
            now = datetime.now(self.ist)
            current_time = now.time()
            
            if self.market_open <= current_time < self.market_close:
                market_close_today = datetime.combine(now.date(), self.market_close)
                market_close_today = self.ist.localize(market_close_today)
                diff = market_close_today - now
                return int(diff.total_seconds() / 60)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error calculating time to market close: {str(e)}")
            return None

class SymbolFormatter:
    def __init__(self):
        self.nse_suffix = '.NS'
        self.bse_suffix = '.BO'
        
        # Common symbol mappings
        self.symbol_mappings = {
            'NIFTY': '^NSEI',
            'SENSEX': '^BSESN',
            'BANKNIFTY': '^NSEBANK',
            'NIFTYIT': '^CNXIT',
            'NIFTYPHARMA': '^CNXPHARMA'
        }

    def format_for_yfinance(self, symbol: str, exchange: str = 'NSE') -> str:
        """Format symbol for Yahoo Finance"""
        try:
            symbol = symbol.upper().strip()
            
            # Check if already formatted
            if symbol.endswith(('.NS', '.BO')) or symbol.startswith('^'):
                return symbol
            
            # Check special mappings
            if symbol in self.symbol_mappings:
                return self.symbol_mappings[symbol]
            
            # Add exchange suffix
            if exchange.upper() == 'BSE':
                return f"{symbol}{self.bse_suffix}"
            else:
                return f"{symbol}{self.nse_suffix}"
                
        except Exception as e:
            logger.error(f"Error formatting symbol {symbol}: {str(e)}")
            return symbol

    def format_for_tradingview(self, symbol: str) -> str:
        """Format symbol for TradingView"""
        try:
            symbol = symbol.upper().strip()
            
            # Remove exchange suffixes
            symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            # Handle index symbols
            if symbol.startswith('^'):
                symbol = symbol[1:]  # Remove ^ prefix
            
            return symbol
            
        except Exception as e:
            logger.error(f"Error formatting symbol for TradingView {symbol}: {str(e)}")
            return symbol

    def get_exchange_from_symbol(self, symbol: str) -> str:
        """Determine exchange from symbol"""
        try:
            if symbol.endswith('.BO'):
                return 'BSE'
            elif symbol.endswith('.NS') or not symbol.endswith(('.NS', '.BO')):
                return 'NSE'
            else:
                return 'NSE'  # Default
                
        except Exception as e:
            logger.error(f"Error determining exchange for {symbol}: {str(e)}")
            return 'NSE'

class BrokerageIntegration:
    """Placeholder for Zerodha/Upstox integration"""
    
    def __init__(self):
        self.zerodha_enabled = False
        self.upstox_enabled = False
        
        # API credentials (set via environment variables)
        self.zerodha_api_key = ""
        self.zerodha_access_token = ""
        self.upstox_api_key = ""
        self.upstox_access_token = ""

    def configure_zerodha(self, api_key: str, access_token: str):
        """Configure Zerodha KiteConnect"""
        self.zerodha_api_key = api_key
        self.zerodha_access_token = access_token
        self.zerodha_enabled = True
        logger.info("Zerodha integration configured")

    def configure_upstox(self, api_key: str, access_token: str):
        """Configure Upstox API"""
        self.upstox_api_key = api_key
        self.upstox_access_token = access_token
        self.upstox_enabled = True
        logger.info("Upstox integration configured")

    def place_order_zerodha(self, symbol: str, quantity: int, order_type: str, 
                           price: float = None) -> Dict[str, Any]:
        """Place order via Zerodha (placeholder)"""
        try:
            if not self.zerodha_enabled:
                return {'error': 'Zerodha not configured'}
            
            # Placeholder for actual KiteConnect integration
            logger.info(f"Zerodha order: {symbol} {quantity} {order_type} {price}")
            
            return {
                'status': 'success',
                'order_id': 'ZER123456',
                'message': 'Order placed successfully'
            }
            
        except Exception as e:
            logger.error(f"Error placing Zerodha order: {str(e)}")
            return {'error': str(e)}

    def place_order_upstox(self, symbol: str, quantity: int, order_type: str, 
                          price: float = None) -> Dict[str, Any]:
        """Place order via Upstox (placeholder)"""
        try:
            if not self.upstox_enabled:
                return {'error': 'Upstox not configured'}
            
            # Placeholder for actual Upstox API integration
            logger.info(f"Upstox order: {symbol} {quantity} {order_type} {price}")
            
            return {
                'status': 'success',
                'order_id': 'UPS123456',
                'message': 'Order placed successfully'
            }
            
        except Exception as e:
            logger.error(f"Error placing Upstox order: {str(e)}")
            return {'error': str(e)}

def format_indian_currency(amount: float) -> str:
    """Format amount in Indian currency format"""
    try:
        if amount >= 10000000:  # 1 crore
            return f"₹{amount/10000000:.2f} Cr"
        elif amount >= 100000:  # 1 lakh
            return f"₹{amount/100000:.2f} L"
        elif amount >= 1000:  # 1 thousand
            return f"₹{amount/1000:.2f} K"
        else:
            return f"₹{amount:.2f}"
    except:
        return f"₹{amount}"

def calculate_lot_size(symbol: str) -> int:
    """Get lot size for F&O stocks (placeholder)"""
    # Common lot sizes (update as needed)
    lot_sizes = {
        'NIFTY': 50,
        'BANKNIFTY': 25,
        'RELIANCE': 250,
        'TCS': 150,
        'HDFCBANK': 550,
        'INFY': 300,
        'ICICIBANK': 1375,
        'SBIN': 3000
    }
    
    return lot_sizes.get(symbol.replace('.NS', '').replace('.BO', ''), 1)