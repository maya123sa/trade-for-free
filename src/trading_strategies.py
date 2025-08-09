"""
Indian trading strategies
Opening Range Breakout (ORB), SuperTrend + EMA crossover, Nifty correlation
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class IndianTradingStrategies:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        self.market_open = time(9, 15)  # 9:15 AM IST
        self.orb_end = time(9, 45)      # 9:45 AM IST (30-minute ORB)
        self.market_close = time(15, 30) # 3:30 PM IST

    def detect_orb_setup(self, data: pd.DataFrame, orb_minutes: int = 30) -> Dict[str, Any]:
        """Detect Opening Range Breakout setup (9:15-9:45 AM IST)"""
        try:
            if len(data) < orb_minutes:
                return {}
            
            # Get today's data (assuming intraday data)
            today = datetime.now(self.ist).date()
            
            # Filter data for ORB period (first 30 minutes)
            orb_data = data.head(orb_minutes)
            
            if orb_data.empty:
                return {}
            
            # Calculate ORB levels
            orb_high = orb_data['High'].max()
            orb_low = orb_data['Low'].min()
            orb_range = orb_high - orb_low
            orb_volume = orb_data['Volume'].sum()
            
            # Current price
            current_price = data['Close'].iloc[-1]
            
            # Determine breakout status
            breakout_status = "NONE"
            if current_price > orb_high:
                breakout_status = "BULLISH_BREAKOUT"
            elif current_price < orb_low:
                breakout_status = "BEARISH_BREAKOUT"
            elif orb_low <= current_price <= orb_high:
                breakout_status = "INSIDE_RANGE"
            
            # Calculate targets (1:2 risk-reward)
            if breakout_status == "BULLISH_BREAKOUT":
                target1 = orb_high + orb_range
                target2 = orb_high + (2 * orb_range)
                stop_loss = orb_low
            elif breakout_status == "BEARISH_BREAKOUT":
                target1 = orb_low - orb_range
                target2 = orb_low - (2 * orb_range)
                stop_loss = orb_high
            else:
                target1 = target2 = stop_loss = None
            
            return {
                'orb_high': round(orb_high, 2),
                'orb_low': round(orb_low, 2),
                'orb_range': round(orb_range, 2),
                'orb_volume': orb_volume,
                'current_price': round(current_price, 2),
                'breakout_status': breakout_status,
                'target1': round(target1, 2) if target1 else None,
                'target2': round(target2, 2) if target2 else None,
                'stop_loss': round(stop_loss, 2) if stop_loss else None,
                'risk_reward': 2.0 if target1 else None
            }
            
        except Exception as e:
            logger.error(f"Error detecting ORB setup: {str(e)}")
            return {}

    def supertrend_ema_crossover(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """SuperTrend + EMA crossover strategy"""
        try:
            if 'supertrend' not in indicators or 'moving_averages' not in indicators:
                return {}
            
            supertrend = indicators['supertrend']
            ma = indicators['moving_averages']
            current_price = indicators['current_price']
            
            # Get EMA 20 value
            ema_20 = ma.get('ema_20')
            if not ema_20:
                return {}
            
            # Strategy conditions
            conditions = {
                'supertrend_bullish': supertrend['direction'] == 'Bullish',
                'price_above_ema': current_price > ema_20,
                'supertrend_signal': supertrend['signal'],
                'ema_trend': 'Bullish' if current_price > ema_20 else 'Bearish'
            }
            
            # Generate signal
            if conditions['supertrend_bullish'] and conditions['price_above_ema']:
                signal = "STRONG_BUY"
                confidence = 90
            elif conditions['supertrend_bullish'] or conditions['price_above_ema']:
                signal = "BUY"
                confidence = 70
            elif not conditions['supertrend_bullish'] and not conditions['price_above_ema']:
                signal = "STRONG_SELL"
                confidence = 90
            else:
                signal = "SELL"
                confidence = 70
            
            return {
                'signal': signal,
                'confidence': confidence,
                'conditions': conditions,
                'entry_price': current_price,
                'stop_loss': supertrend['value'],
                'target': current_price * 1.02 if signal in ['BUY', 'STRONG_BUY'] else current_price * 0.98
            }
            
        except Exception as e:
            logger.error(f"Error in SuperTrend EMA crossover: {str(e)}")
            return {}

    def calculate_nifty_correlation(self, stock_data: pd.DataFrame, nifty_data: pd.DataFrame) -> float:
        """Calculate correlation with Nifty 50"""
        try:
            # Align data by dates
            stock_returns = stock_data['Close'].pct_change().dropna()
            nifty_returns = nifty_data['Close'].pct_change().dropna()
            
            # Find common dates
            common_dates = stock_returns.index.intersection(nifty_returns.index)
            
            if len(common_dates) < 10:  # Need at least 10 data points
                return 0.0
            
            stock_aligned = stock_returns.loc[common_dates]
            nifty_aligned = nifty_returns.loc[common_dates]
            
            correlation = stock_aligned.corr(nifty_aligned)
            
            return round(correlation, 3) if not pd.isna(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Nifty correlation: {str(e)}")
            return 0.0

    def scan_orb_opportunities(self) -> List[Dict[str, Any]]:
        """Scan for ORB opportunities across multiple stocks"""
        try:
            from .data_fetcher import IndianStockDataFetcher
            
            data_fetcher = IndianStockDataFetcher()
            nifty_50_stocks = data_fetcher.get_nifty_50_stocks()
            
            orb_opportunities = []
            
            # Check current time
            current_time = datetime.now(self.ist).time()
            
            # Only scan during market hours
            if not (self.market_open <= current_time <= self.market_close):
                return []
            
            # Scan top 20 stocks for performance
            for symbol in nifty_50_stocks[:20]:
                try:
                    # Get intraday data
                    data = data_fetcher.get_stock_data(symbol, interval='5m', period='1d')
                    
                    if data.empty:
                        continue
                    
                    # Detect ORB setup
                    orb_setup = self.detect_orb_setup(data)
                    
                    if orb_setup and orb_setup.get('breakout_status') != 'NONE':
                        orb_setup['symbol'] = symbol
                        orb_opportunities.append(orb_setup)
                        
                except Exception as e:
                    logger.error(f"Error scanning ORB for {symbol}: {str(e)}")
                    continue
            
            # Sort by breakout strength (range and volume)
            orb_opportunities.sort(key=lambda x: x.get('orb_range', 0) * x.get('orb_volume', 0), reverse=True)
            
            return orb_opportunities[:10]  # Return top 10 opportunities
            
        except Exception as e:
            logger.error(f"Error scanning ORB opportunities: {str(e)}")
            return []

    def generate_signals(self, data: pd.DataFrame, indicators: Dict) -> Dict[str, Any]:
        """Generate comprehensive trading signals"""
        try:
            signals = {}
            
            # ORB Signal
            orb_signal = self.detect_orb_setup(data)
            if orb_signal:
                signals['orb'] = orb_signal
            
            # SuperTrend + EMA Signal
            st_ema_signal = self.supertrend_ema_crossover(data, indicators)
            if st_ema_signal:
                signals['supertrend_ema'] = st_ema_signal
            
            # Pivot Point Signals
            if 'pivot_points' in indicators:
                pivot_signals = self._analyze_pivot_signals(indicators['current_price'], 
                                                          indicators['pivot_points'])
                signals['pivot_points'] = pivot_signals
            
            # Volume Analysis
            volume_signal = self._analyze_volume(data)
            if volume_signal:
                signals['volume'] = volume_signal
            
            # Overall Signal Strength
            signals['overall'] = self._calculate_overall_signal(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return {}

    def _analyze_pivot_signals(self, current_price: float, pivot_points: Dict) -> Dict[str, Any]:
        """Analyze pivot point signals"""
        try:
            pp = pivot_points.get('PP', 0)
            r1 = pivot_points.get('R1', 0)
            s1 = pivot_points.get('S1', 0)
            
            if current_price > r1:
                signal = "BULLISH"
                level = "Above R1"
            elif current_price > pp:
                signal = "BULLISH"
                level = "Above Pivot"
            elif current_price < s1:
                signal = "BEARISH"
                level = "Below S1"
            elif current_price < pp:
                signal = "BEARISH"
                level = "Below Pivot"
            else:
                signal = "NEUTRAL"
                level = "At Pivot"
            
            return {
                'signal': signal,
                'level': level,
                'current_price': current_price,
                'pivot': pp,
                'resistance': r1,
                'support': s1
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pivot signals: {str(e)}")
            return {}

    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            if len(data) < 20:
                return {}
            
            current_volume = data['Volume'].iloc[-1]
            avg_volume_20 = data['Volume'].tail(20).mean()
            volume_ratio = current_volume / avg_volume_20
            
            if volume_ratio > 2.0:
                signal = "HIGH_VOLUME"
                strength = "Strong"
            elif volume_ratio > 1.5:
                signal = "ABOVE_AVERAGE"
                strength = "Medium"
            elif volume_ratio < 0.5:
                signal = "LOW_VOLUME"
                strength = "Weak"
            else:
                signal = "NORMAL"
                strength = "Normal"
            
            return {
                'signal': signal,
                'strength': strength,
                'current_volume': int(current_volume),
                'avg_volume': int(avg_volume_20),
                'volume_ratio': round(volume_ratio, 2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {str(e)}")
            return {}

    def _calculate_overall_signal(self, signals: Dict) -> Dict[str, Any]:
        """Calculate overall signal strength"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            
            # Count signals
            for signal_type, signal_data in signals.items():
                if signal_type == 'overall':
                    continue
                
                total_signals += 1
                
                if signal_type == 'orb':
                    if signal_data.get('breakout_status') == 'BULLISH_BREAKOUT':
                        bullish_signals += 1
                    elif signal_data.get('breakout_status') == 'BEARISH_BREAKOUT':
                        bearish_signals += 1
                
                elif signal_type == 'supertrend_ema':
                    if 'BUY' in signal_data.get('signal', ''):
                        bullish_signals += 1
                    elif 'SELL' in signal_data.get('signal', ''):
                        bearish_signals += 1
                
                elif signal_type == 'pivot_points':
                    if signal_data.get('signal') == 'BULLISH':
                        bullish_signals += 1
                    elif signal_data.get('signal') == 'BEARISH':
                        bearish_signals += 1
                
                elif signal_type == 'volume':
                    if signal_data.get('signal') in ['HIGH_VOLUME', 'ABOVE_AVERAGE']:
                        bullish_signals += 0.5  # Volume is supportive but not directional
            
            # Calculate overall signal
            if total_signals == 0:
                return {'signal': 'NO_SIGNAL', 'confidence': 0}
            
            bullish_ratio = bullish_signals / total_signals
            bearish_ratio = bearish_signals / total_signals
            
            if bullish_ratio > 0.7:
                overall_signal = 'STRONG_BUY'
                confidence = int(bullish_ratio * 100)
            elif bullish_ratio > 0.5:
                overall_signal = 'BUY'
                confidence = int(bullish_ratio * 100)
            elif bearish_ratio > 0.7:
                overall_signal = 'STRONG_SELL'
                confidence = int(bearish_ratio * 100)
            elif bearish_ratio > 0.5:
                overall_signal = 'SELL'
                confidence = int(bearish_ratio * 100)
            else:
                overall_signal = 'NEUTRAL'
                confidence = 50
            
            return {
                'signal': overall_signal,
                'confidence': confidence,
                'bullish_signals': bullish_signals,
                'bearish_signals': bearish_signals,
                'total_signals': total_signals
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall signal: {str(e)}")
            return {'signal': 'ERROR', 'confidence': 0}