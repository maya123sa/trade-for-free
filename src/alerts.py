"""
Alert management for Telegram/WhatsApp notifications
Bilingual support (English/Hindi)
"""

import requests
import json
from datetime import datetime
import pytz
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        
        # Telegram configuration (set these in environment variables)
        self.telegram_bot_token = ""  # Set your bot token
        self.telegram_chat_id = ""    # Set your chat ID
        
        # WhatsApp configuration (using Twilio or similar service)
        self.whatsapp_enabled = False
        
        # Language settings
        self.languages = ['en', 'hi']  # English and Hindi
        
        # Hindi translations
        self.translations = {
            'hi': {
                'buy_signal': 'खरीदारी संकेत',
                'sell_signal': 'बिक्री संकेत',
                'strong_buy': 'मजबूत खरीदारी',
                'strong_sell': 'मजबूत बिक्री',
                'neutral': 'तटस्थ',
                'current_price': 'वर्तमान मूल्य',
                'target': 'लक्ष्य',
                'stop_loss': 'स्टॉप लॉस',
                'confidence': 'विश्वास',
                'market_open': 'बाजार खुला',
                'market_closed': 'बाजार बंद',
                'orb_breakout': 'ओआरबी ब्रेकआउट',
                'supertrend_signal': 'सुपरट्रेंड संकेत',
                'volume_spike': 'वॉल्यूम स्पाइक'
            }
        }

    def send_telegram_alert(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send alert to Telegram"""
        try:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                logger.warning("Telegram credentials not configured")
                return False
            
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully")
                return True
            else:
                logger.error(f"Failed to send Telegram alert: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {str(e)}")
            return False

    def send_whatsapp_alert(self, message: str, phone_number: str) -> bool:
        """Send alert to WhatsApp (placeholder for Twilio integration)"""
        try:
            if not self.whatsapp_enabled:
                logger.info("WhatsApp alerts not enabled")
                return False
            
            # Implement WhatsApp integration using Twilio or similar service
            # This is a placeholder implementation
            
            logger.info(f"WhatsApp alert would be sent to {phone_number}: {message}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending WhatsApp alert: {str(e)}")
            return False

    def format_trading_signal_alert(self, symbol: str, signals: Dict, language: str = 'en') -> str:
        """Format trading signal alert message"""
        try:
            current_time = datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S IST')
            
            if language == 'hi':
                t = self.translations['hi']
            else:
                t = {
                    'buy_signal': 'BUY Signal',
                    'sell_signal': 'SELL Signal',
                    'strong_buy': 'STRONG BUY',
                    'strong_sell': 'STRONG SELL',
                    'neutral': 'NEUTRAL',
                    'current_price': 'Current Price',
                    'target': 'Target',
                    'stop_loss': 'Stop Loss',
                    'confidence': 'Confidence',
                    'market_open': 'Market Open',
                    'market_closed': 'Market Closed',
                    'orb_breakout': 'ORB Breakout',
                    'supertrend_signal': 'SuperTrend Signal',
                    'volume_spike': 'Volume Spike'
                }
            
            # Overall signal
            overall = signals.get('overall', {})
            signal_type = overall.get('signal', 'NEUTRAL')
            confidence = overall.get('confidence', 0)
            
            # Translate signal type
            if 'BUY' in signal_type:
                signal_emoji = '🟢'
                signal_text = t['strong_buy'] if 'STRONG' in signal_type else t['buy_signal']
            elif 'SELL' in signal_type:
                signal_emoji = '🔴'
                signal_text = t['strong_sell'] if 'STRONG' in signal_type else t['sell_signal']
            else:
                signal_emoji = '🟡'
                signal_text = t['neutral']
            
            message = f"""
🚨 <b>{symbol} Trading Alert</b> 🚨

{signal_emoji} <b>{signal_text}</b>
📊 {t['confidence']}: {confidence}%
🕐 {current_time}

"""
            
            # Add specific signals
            if 'orb' in signals:
                orb = signals['orb']
                if orb.get('breakout_status') != 'NONE':
                    message += f"📈 {t['orb_breakout']}: {orb.get('breakout_status', 'N/A')}\n"
                    if orb.get('target1'):
                        message += f"🎯 {t['target']}: {orb['target1']}\n"
                    if orb.get('stop_loss'):
                        message += f"🛑 {t['stop_loss']}: {orb['stop_loss']}\n"
            
            if 'supertrend_ema' in signals:
                st_ema = signals['supertrend_ema']
                message += f"📊 {t['supertrend_signal']}: {st_ema.get('signal', 'N/A')}\n"
            
            if 'volume' in signals:
                volume = signals['volume']
                if volume.get('signal') in ['HIGH_VOLUME', 'ABOVE_AVERAGE']:
                    message += f"📊 {t['volume_spike']}: {volume.get('volume_ratio', 0)}x\n"
            
            message += f"\n⚠️ This is not financial advice. Trade at your own risk."
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting trading signal alert: {str(e)}")
            return f"Error formatting alert for {symbol}"

    def format_orb_alert(self, orb_opportunities: List[Dict], language: str = 'en') -> str:
        """Format ORB opportunities alert"""
        try:
            current_time = datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S IST')
            
            if not orb_opportunities:
                return f"🔍 No ORB opportunities found at {current_time}"
            
            message = f"""
🚨 <b>ORB Scanner Alert</b> 🚨
🕐 {current_time}

<b>Top ORB Opportunities:</b>

"""
            
            for i, orb in enumerate(orb_opportunities[:5], 1):
                symbol = orb.get('symbol', 'N/A')
                status = orb.get('breakout_status', 'N/A')
                current_price = orb.get('current_price', 0)
                orb_range = orb.get('orb_range', 0)
                
                if status == 'BULLISH_BREAKOUT':
                    emoji = '🟢'
                elif status == 'BEARISH_BREAKOUT':
                    emoji = '🔴'
                else:
                    emoji = '🟡'
                
                message += f"{i}. {emoji} <b>{symbol}</b>\n"
                message += f"   Status: {status}\n"
                message += f"   Price: ₹{current_price}\n"
                message += f"   Range: ₹{orb_range}\n\n"
            
            message += "⚠️ This is not financial advice. Trade at your own risk."
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting ORB alert: {str(e)}")
            return "Error formatting ORB alert"

    def format_sector_alert(self, sector_data: Dict, language: str = 'en') -> str:
        """Format sector performance alert"""
        try:
            current_time = datetime.now(self.ist).strftime('%Y-%m-%d %H:%M:%S IST')
            
            if not sector_data or 'sectors' not in sector_data:
                return f"📊 No sector data available at {current_time}"
            
            sectors = sector_data['sectors']
            best_sector = sector_data.get('best_sector', 'N/A')
            worst_sector = sector_data.get('worst_sector', 'N/A')
            
            message = f"""
📊 <b>Sector Performance Alert</b> 📊
🕐 {current_time}

🏆 <b>Best Sector:</b> {best_sector}
📉 <b>Worst Sector:</b> {worst_sector}

<b>Top Performers:</b>
"""
            
            for sector in sectors[:5]:
                name = sector['sector']
                change = sector['change_pct']
                
                if change > 0:
                    emoji = '🟢'
                    sign = '+'
                else:
                    emoji = '🔴'
                    sign = ''
                
                message += f"{emoji} {name}: {sign}{change}%\n"
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting sector alert: {str(e)}")
            return "Error formatting sector alert"

    def send_trading_alert(self, symbol: str, signals: Dict, language: str = 'en') -> bool:
        """Send comprehensive trading alert"""
        try:
            message = self.format_trading_signal_alert(symbol, signals, language)
            
            # Send to Telegram
            telegram_sent = self.send_telegram_alert(message)
            
            # Send to WhatsApp (if configured)
            whatsapp_sent = True  # Placeholder
            
            return telegram_sent or whatsapp_sent
            
        except Exception as e:
            logger.error(f"Error sending trading alert: {str(e)}")
            return False

    def send_orb_scanner_alert(self, orb_opportunities: List[Dict], language: str = 'en') -> bool:
        """Send ORB scanner alert"""
        try:
            message = self.format_orb_alert(orb_opportunities, language)
            return self.send_telegram_alert(message)
            
        except Exception as e:
            logger.error(f"Error sending ORB scanner alert: {str(e)}")
            return False

    def send_sector_performance_alert(self, sector_data: Dict, language: str = 'en') -> bool:
        """Send sector performance alert"""
        try:
            message = self.format_sector_alert(sector_data, language)
            return self.send_telegram_alert(message)
            
        except Exception as e:
            logger.error(f"Error sending sector performance alert: {str(e)}")
            return False

    def configure_telegram(self, bot_token: str, chat_id: str):
        """Configure Telegram credentials"""
        self.telegram_bot_token = bot_token
        self.telegram_chat_id = chat_id
        logger.info("Telegram configuration updated")

    def configure_whatsapp(self, enabled: bool = True):
        """Configure WhatsApp alerts"""
        self.whatsapp_enabled = enabled
        logger.info(f"WhatsApp alerts {'enabled' if enabled else 'disabled'}")