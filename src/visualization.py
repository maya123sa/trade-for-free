"""
Visualization module for Indian stock analysis
Charts, heatmaps, and market hour annotations
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class IndianStockVisualizer:
    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')
        plt.style.use('seaborn-v0_8')
        
        # Indian market hours
        self.pre_open = time(9, 0)   # 9:00 AM IST
        self.market_open = time(9, 15) # 9:15 AM IST
        self.market_close = time(15, 30) # 3:30 PM IST
        
        # Color scheme
        self.colors = {
            'bullish': '#00C851',
            'bearish': '#FF4444',
            'neutral': '#FFA500',
            'background': '#F8F9FA',
            'grid': '#E0E0E0'
        }

    def plot_candlestick_with_indicators(self, data: pd.DataFrame, indicators: Dict, 
                                       symbol: str) -> str:
        """Create candlestick chart with Indian indicators"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), 
                                              gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Main candlestick chart
            self._plot_candlesticks(ax1, data)
            
            # Add SuperTrend
            if 'supertrend' in indicators:
                self._add_supertrend(ax1, data, indicators['supertrend'])
            
            # Add moving averages
            if 'moving_averages' in indicators:
                self._add_moving_averages(ax1, data, indicators['moving_averages'])
            
            # Add pivot points
            if 'pivot_points' in indicators:
                self._add_pivot_points(ax1, indicators['pivot_points'])
            
            # Volume chart
            self._plot_volume(ax2, data)
            
            # Volume profile
            if 'volume_profile' in indicators:
                self._plot_volume_profile(ax3, indicators['volume_profile'])
            
            # Market hours annotation
            self._annotate_market_hours(ax1)
            
            # Formatting
            ax1.set_title(f'{symbol} - Technical Analysis (IST)', fontsize=16, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax2.set_ylabel('Volume')
            ax3.set_ylabel('Volume Profile')
            
            plt.tight_layout()
            
            # Save to base64
            import io
            import base64
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {str(e)}")
            return ""

    def _plot_candlesticks(self, ax, data: pd.DataFrame):
        """Plot candlestick chart"""
        try:
            from matplotlib.patches import Rectangle
            
            for i, (idx, row) in enumerate(data.iterrows()):
                color = self.colors['bullish'] if row['Close'] > row['Open'] else self.colors['bearish']
                
                # High-Low line
                ax.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1)
                
                # Body rectangle
                height = abs(row['Close'] - row['Open'])
                bottom = min(row['Open'], row['Close'])
                
                rect = Rectangle((i-0.3, bottom), 0.6, height, 
                               facecolor=color, edgecolor='black', alpha=0.8)
                ax.add_patch(rect)
            
            ax.set_xlim(-0.5, len(data)-0.5)
            
        except Exception as e:
            logger.error(f"Error plotting candlesticks: {str(e)}")

    def _add_supertrend(self, ax, data: pd.DataFrame, supertrend_data: Dict):
        """Add SuperTrend indicator"""
        try:
            # This would require the actual SuperTrend values from the data
            # For now, just add a placeholder line
            ax.axhline(y=supertrend_data['value'], color='purple', 
                      linestyle='--', label=f"SuperTrend: {supertrend_data['value']}")
            
        except Exception as e:
            logger.error(f"Error adding SuperTrend: {str(e)}")

    def _add_moving_averages(self, ax, data: pd.DataFrame, ma_data: Dict):
        """Add moving averages"""
        try:
            x = range(len(data))
            
            if ma_data.get('ema_20'):
                ax.axhline(y=ma_data['ema_20'], color='blue', 
                          linestyle='-', alpha=0.7, label=f"EMA 20: {ma_data['ema_20']}")
            
            if ma_data.get('sma_50'):
                ax.axhline(y=ma_data['sma_50'], color='orange', 
                          linestyle='-', alpha=0.7, label=f"SMA 50: {ma_data['sma_50']}")
            
            if ma_data.get('sma_200'):
                ax.axhline(y=ma_data['sma_200'], color='red', 
                          linestyle='-', alpha=0.7, label=f"SMA 200: {ma_data['sma_200']}")
            
            ax.legend()
            
        except Exception as e:
            logger.error(f"Error adding moving averages: {str(e)}")

    def _add_pivot_points(self, ax, pivot_data: Dict):
        """Add pivot points"""
        try:
            colors = ['red', 'orange', 'yellow', 'green', 'yellow', 'orange', 'red']
            levels = ['R3', 'R2', 'R1', 'PP', 'S1', 'S2', 'S3']
            
            for level, color in zip(levels, colors):
                if level in pivot_data:
                    ax.axhline(y=pivot_data[level], color=color, 
                              linestyle=':', alpha=0.6, label=f"{level}: {pivot_data[level]}")
            
        except Exception as e:
            logger.error(f"Error adding pivot points: {str(e)}")

    def _plot_volume(self, ax, data: pd.DataFrame):
        """Plot volume bars"""
        try:
            colors = [self.colors['bullish'] if close > open_price else self.colors['bearish'] 
                     for close, open_price in zip(data['Close'], data['Open'])]
            
            ax.bar(range(len(data)), data['Volume'], color=colors, alpha=0.7)
            ax.set_ylabel('Volume')
            
        except Exception as e:
            logger.error(f"Error plotting volume: {str(e)}")

    def _plot_volume_profile(self, ax, vp_data: Dict):
        """Plot volume profile"""
        try:
            if 'volume_profile' in vp_data and 'price_bins' in vp_data:
                volume_profile = vp_data['volume_profile']
                price_bins = vp_data['price_bins']
                
                # Create horizontal bar chart
                y_pos = [(price_bins[i] + price_bins[i+1]) / 2 for i in range(len(volume_profile))]
                ax.barh(y_pos, volume_profile, alpha=0.7, color='gray')
                
                # Highlight POC, VAH, VAL
                if 'POC' in vp_data:
                    ax.axhline(y=vp_data['POC'], color='red', linestyle='-', 
                              linewidth=2, label=f"POC: {vp_data['POC']}")
                if 'VAH' in vp_data:
                    ax.axhline(y=vp_data['VAH'], color='green', linestyle='--', 
                              label=f"VAH: {vp_data['VAH']}")
                if 'VAL' in vp_data:
                    ax.axhline(y=vp_data['VAL'], color='green', linestyle='--', 
                              label=f"VAL: {vp_data['VAL']}")
                
                ax.legend()
            
        except Exception as e:
            logger.error(f"Error plotting volume profile: {str(e)}")

    def _annotate_market_hours(self, ax):
        """Annotate Indian market hours"""
        try:
            # Add market hours text
            ax.text(0.02, 0.98, 'Market Hours (IST):\nPre-open: 9:00-9:15\nTrading: 9:15-15:30', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        except Exception as e:
            logger.error(f"Error annotating market hours: {str(e)}")

    def generate_sector_heatmap(self) -> Dict[str, Any]:
        """Generate NSE sector heatmap"""
        try:
            from .data_fetcher import IndianStockDataFetcher
            
            data_fetcher = IndianStockDataFetcher()
            sector_data = data_fetcher.get_sector_performance()
            
            if not sector_data:
                return {}
            
            # Prepare data for heatmap
            sectors = list(sector_data.keys())
            changes = [sector_data[sector]['avg_change'] for sector in sectors]
            
            # Create heatmap data
            heatmap_data = []
            for sector, change in zip(sectors, changes):
                heatmap_data.append({
                    'sector': sector,
                    'change_pct': change,
                    'color': self._get_heatmap_color(change),
                    'stocks': sector_data[sector]['stocks']
                })
            
            # Sort by performance
            heatmap_data.sort(key=lambda x: x['change_pct'], reverse=True)
            
            return {
                'sectors': heatmap_data,
                'timestamp': datetime.now(self.ist).isoformat(),
                'best_sector': heatmap_data[0]['sector'] if heatmap_data else None,
                'worst_sector': heatmap_data[-1]['sector'] if heatmap_data else None
            }
            
        except Exception as e:
            logger.error(f"Error generating sector heatmap: {str(e)}")
            return {}

    def _get_heatmap_color(self, change_pct: float) -> str:
        """Get color for heatmap based on percentage change"""
        if change_pct > 2:
            return '#00C851'  # Strong green
        elif change_pct > 1:
            return '#4CAF50'  # Green
        elif change_pct > 0:
            return '#8BC34A'  # Light green
        elif change_pct > -1:
            return '#FFC107'  # Yellow
        elif change_pct > -2:
            return '#FF9800'  # Orange
        else:
            return '#F44336'  # Red

    def create_orb_visualization(self, orb_data: Dict, symbol: str) -> str:
        """Create ORB (Opening Range Breakout) visualization"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot ORB levels
            orb_high = orb_data.get('orb_high', 0)
            orb_low = orb_data.get('orb_low', 0)
            current_price = orb_data.get('current_price', 0)
            
            # ORB range
            ax.axhspan(orb_low, orb_high, alpha=0.3, color='gray', label='ORB Range')
            ax.axhline(y=orb_high, color='red', linestyle='-', linewidth=2, label=f'ORB High: {orb_high}')
            ax.axhline(y=orb_low, color='green', linestyle='-', linewidth=2, label=f'ORB Low: {orb_low}')
            
            # Current price
            ax.axhline(y=current_price, color='blue', linestyle='--', linewidth=2, 
                      label=f'Current: {current_price}')
            
            # Targets and stop loss
            if orb_data.get('target1'):
                ax.axhline(y=orb_data['target1'], color='orange', linestyle=':', 
                          label=f"Target 1: {orb_data['target1']}")
            if orb_data.get('target2'):
                ax.axhline(y=orb_data['target2'], color='purple', linestyle=':', 
                          label=f"Target 2: {orb_data['target2']}")
            if orb_data.get('stop_loss'):
                ax.axhline(y=orb_data['stop_loss'], color='red', linestyle=':', 
                          label=f"Stop Loss: {orb_data['stop_loss']}")
            
            # Formatting
            ax.set_title(f'{symbol} - Opening Range Breakout (ORB)\nStatus: {orb_data.get("breakout_status", "N/A")}', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add time annotation
            ax.text(0.02, 0.98, 'ORB Period: 9:15-9:45 AM IST', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Convert to base64
            import io
            import base64
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating ORB visualization: {str(e)}")
            return ""

    def create_option_chain_visualization(self, option_data: Dict) -> str:
        """Create option chain PCR/OI visualization"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            
            calls = option_data.get('calls', [])
            puts = option_data.get('puts', [])
            
            if not calls or not puts:
                return ""
            
            # OI Chart
            call_strikes = [c['strike'] for c in calls]
            call_oi = [c['oi'] for c in calls]
            put_strikes = [p['strike'] for p in puts]
            put_oi = [p['oi'] for p in puts]
            
            ax1.bar([str(s) for s in call_strikes], call_oi, alpha=0.7, 
                   color=self.colors['bullish'], label='Call OI')
            ax1.bar([str(s) for s in put_strikes], [-oi for oi in put_oi], alpha=0.7, 
                   color=self.colors['bearish'], label='Put OI')
            
            ax1.set_title('Open Interest by Strike')
            ax1.set_xlabel('Strike Price')
            ax1.set_ylabel('Open Interest')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume Chart
            call_volume = [c['volume'] for c in calls]
            put_volume = [p['volume'] for p in puts]
            
            ax2.bar([str(s) for s in call_strikes], call_volume, alpha=0.7, 
                   color=self.colors['bullish'], label='Call Volume')
            ax2.bar([str(s) for s in put_strikes], [-v for v in put_volume], alpha=0.7, 
                   color=self.colors['bearish'], label='Put Volume')
            
            ax2.set_title('Volume by Strike')
            ax2.set_xlabel('Strike Price')
            ax2.set_ylabel('Volume')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64
            import io
            import base64
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating option chain visualization: {str(e)}")
            return ""