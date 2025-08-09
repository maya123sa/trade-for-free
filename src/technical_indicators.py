"""
Indian-specific technical indicators
SuperTrend, Heikin-Ashi, Pivot Points, Volume Profile
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class IndianTechnicalIndicators:
    def __init__(self):
        pass

    def calculate_supertrend(self, data: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate SuperTrend indicator (popular in Indian markets)"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # Calculate ATR
            atr = talib.ATR(high, low, close, timeperiod=period)
            
            # Calculate basic bands
            hl2 = (high + low) / 2
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # Initialize SuperTrend
            supertrend = pd.Series(index=data.index, dtype=float)
            direction = pd.Series(index=data.index, dtype=int)
            
            for i in range(1, len(data)):
                # Upper band calculation
                if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
                
                # Lower band calculation
                if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                
                # SuperTrend calculation
                if i == 1:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    if supertrend.iloc[i-1] == upper_band.iloc[i-1] and close.iloc[i] <= upper_band.iloc[i]:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1
                    elif supertrend.iloc[i-1] == upper_band.iloc[i-1] and close.iloc[i] > upper_band.iloc[i]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
                    elif supertrend.iloc[i-1] == lower_band.iloc[i-1] and close.iloc[i] >= lower_band.iloc[i]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
                    elif supertrend.iloc[i-1] == lower_band.iloc[i-1] and close.iloc[i] < lower_band.iloc[i]:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1
            
            result = data.copy()
            result['SuperTrend'] = supertrend
            result['ST_Direction'] = direction
            result['ST_UpperBand'] = upper_band
            result['ST_LowerBand'] = lower_band
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {str(e)}")
            return data

    def calculate_heikin_ashi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin-Ashi candles"""
        try:
            ha_data = data.copy()
            
            # Initialize first Heikin-Ashi candle
            ha_data.loc[ha_data.index[0], 'HA_Close'] = (data['Open'][0] + data['High'][0] + 
                                                        data['Low'][0] + data['Close'][0]) / 4
            ha_data.loc[ha_data.index[0], 'HA_Open'] = (data['Open'][0] + data['Close'][0]) / 2
            ha_data.loc[ha_data.index[0], 'HA_High'] = data['High'][0]
            ha_data.loc[ha_data.index[0], 'HA_Low'] = data['Low'][0]
            
            # Calculate subsequent Heikin-Ashi candles
            for i in range(1, len(data)):
                idx = ha_data.index[i]
                prev_idx = ha_data.index[i-1]
                
                # HA Close
                ha_data.loc[idx, 'HA_Close'] = (data.loc[idx, 'Open'] + data.loc[idx, 'High'] + 
                                               data.loc[idx, 'Low'] + data.loc[idx, 'Close']) / 4
                
                # HA Open
                ha_data.loc[idx, 'HA_Open'] = (ha_data.loc[prev_idx, 'HA_Open'] + 
                                              ha_data.loc[prev_idx, 'HA_Close']) / 2
                
                # HA High and Low
                ha_data.loc[idx, 'HA_High'] = max(data.loc[idx, 'High'], 
                                                 ha_data.loc[idx, 'HA_Open'], 
                                                 ha_data.loc[idx, 'HA_Close'])
                ha_data.loc[idx, 'HA_Low'] = min(data.loc[idx, 'Low'], 
                                                ha_data.loc[idx, 'HA_Open'], 
                                                ha_data.loc[idx, 'HA_Close'])
            
            return ha_data
            
        except Exception as e:
            logger.error(f"Error calculating Heikin-Ashi: {str(e)}")
            return data

    def calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate classic pivot points"""
        try:
            # Use previous day's data for pivot calculation
            prev_high = data['High'].iloc[-2]
            prev_low = data['Low'].iloc[-2]
            prev_close = data['Close'].iloc[-2]
            
            # Pivot Point
            pp = (prev_high + prev_low + prev_close) / 3
            
            # Support and Resistance levels
            r1 = 2 * pp - prev_low
            s1 = 2 * pp - prev_high
            r2 = pp + (prev_high - prev_low)
            s2 = pp - (prev_high - prev_low)
            r3 = prev_high + 2 * (pp - prev_low)
            s3 = prev_low - 2 * (prev_high - pp)
            
            return {
                'PP': round(pp, 2),
                'R1': round(r1, 2),
                'R2': round(r2, 2),
                'R3': round(r3, 2),
                'S1': round(s1, 2),
                'S2': round(s2, 2),
                'S3': round(s3, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating pivot points: {str(e)}")
            return {}

    def calculate_volume_profile(self, data: pd.DataFrame, bins: int = 20) -> Dict[str, Any]:
        """Calculate Volume Profile with POC, VAH, VAL"""
        try:
            # Create price bins
            price_min = data['Low'].min()
            price_max = data['High'].max()
            price_bins = np.linspace(price_min, price_max, bins + 1)
            
            # Calculate volume for each price level
            volume_profile = np.zeros(bins)
            
            for i in range(len(data)):
                # Find which bin this candle's price range falls into
                low_bin = np.digitize(data['Low'].iloc[i], price_bins) - 1
                high_bin = np.digitize(data['High'].iloc[i], price_bins) - 1
                
                # Distribute volume across the price range
                volume_per_bin = data['Volume'].iloc[i] / (high_bin - low_bin + 1)
                for bin_idx in range(max(0, low_bin), min(bins, high_bin + 1)):
                    volume_profile[bin_idx] += volume_per_bin
            
            # Find Point of Control (POC) - highest volume price level
            poc_idx = np.argmax(volume_profile)
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            
            # Calculate Value Area (70% of volume)
            total_volume = np.sum(volume_profile)
            value_area_volume = total_volume * 0.7
            
            # Find Value Area High (VAH) and Value Area Low (VAL)
            sorted_indices = np.argsort(volume_profile)[::-1]
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in sorted_indices:
                cumulative_volume += volume_profile[idx]
                value_area_indices.append(idx)
                if cumulative_volume >= value_area_volume:
                    break
            
            vah_idx = max(value_area_indices)
            val_idx = min(value_area_indices)
            
            vah_price = (price_bins[vah_idx] + price_bins[vah_idx + 1]) / 2
            val_price = (price_bins[val_idx] + price_bins[val_idx + 1]) / 2
            
            return {
                'POC': round(poc_price, 2),
                'VAH': round(vah_price, 2),
                'VAL': round(val_price, 2),
                'volume_profile': volume_profile.tolist(),
                'price_bins': price_bins.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return {}

    def calculate_ema_sma(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA(20) and SMA(50, 200)"""
        try:
            result = data.copy()
            
            # EMA 20
            result['EMA_20'] = talib.EMA(data['Close'], timeperiod=20)
            
            # SMA 50 and 200
            result['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
            result['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating EMA/SMA: {str(e)}")
            return data

    def analyze_option_chain(self, option_data: Dict) -> Dict[str, Any]:
        """Analyze option chain for PCR and OI analysis"""
        try:
            if not option_data or 'calls' not in option_data or 'puts' not in option_data:
                return {}
            
            calls = option_data['calls']
            puts = option_data['puts']
            
            # Calculate Put-Call Ratio (PCR)
            total_call_oi = sum([call['oi'] for call in calls])
            total_put_oi = sum([put['oi'] for put in puts])
            pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            
            total_call_volume = sum([call['volume'] for call in calls])
            total_put_volume = sum([put['volume'] for put in puts])
            pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 0
            
            # Find max pain (strike with highest total OI)
            strike_oi = {}
            for call in calls:
                strike = call['strike']
                if strike not in strike_oi:
                    strike_oi[strike] = 0
                strike_oi[strike] += call['oi']
            
            for put in puts:
                strike = put['strike']
                if strike not in strike_oi:
                    strike_oi[strike] = 0
                strike_oi[strike] += put['oi']
            
            max_pain_strike = max(strike_oi.keys(), key=lambda k: strike_oi[k])
            
            # Support and resistance from OI
            call_oi_by_strike = {call['strike']: call['oi'] for call in calls}
            put_oi_by_strike = {put['strike']: put['oi'] for put in puts}
            
            resistance_levels = sorted(call_oi_by_strike.keys(), 
                                     key=lambda k: call_oi_by_strike[k], reverse=True)[:3]
            support_levels = sorted(put_oi_by_strike.keys(), 
                                  key=lambda k: put_oi_by_strike[k], reverse=True)[:3]
            
            return {
                'pcr_oi': round(pcr_oi, 3),
                'pcr_volume': round(pcr_volume, 3),
                'max_pain': max_pain_strike,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'resistance_levels': resistance_levels,
                'support_levels': support_levels,
                'interpretation': self._interpret_pcr(pcr_oi)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing option chain: {str(e)}")
            return {}

    def _interpret_pcr(self, pcr: float) -> str:
        """Interpret PCR values"""
        if pcr > 1.5:
            return "Extremely Bullish (High Put OI)"
        elif pcr > 1.2:
            return "Bullish (More Put OI)"
        elif pcr > 0.8:
            return "Neutral"
        elif pcr > 0.6:
            return "Bearish (More Call OI)"
        else:
            return "Extremely Bearish (High Call OI)"

    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all Indian technical indicators"""
        try:
            # SuperTrend
            data_with_st = self.calculate_supertrend(data)
            
            # Heikin-Ashi
            data_with_ha = self.calculate_heikin_ashi(data_with_st)
            
            # EMA/SMA
            data_with_ma = self.calculate_ema_sma(data_with_ha)
            
            # Pivot Points
            pivot_points = self.calculate_pivot_points(data)
            
            # Volume Profile
            volume_profile = self.calculate_volume_profile(data)
            
            # Current values
            current_price = data['Close'].iloc[-1]
            supertrend_value = data_with_st['SuperTrend'].iloc[-1]
            st_direction = data_with_st['ST_Direction'].iloc[-1]
            
            return {
                'current_price': round(current_price, 2),
                'supertrend': {
                    'value': round(supertrend_value, 2),
                    'direction': 'Bullish' if st_direction == 1 else 'Bearish',
                    'signal': 'BUY' if current_price > supertrend_value else 'SELL'
                },
                'heikin_ashi': {
                    'open': round(data_with_ha['HA_Open'].iloc[-1], 2),
                    'high': round(data_with_ha['HA_High'].iloc[-1], 2),
                    'low': round(data_with_ha['HA_Low'].iloc[-1], 2),
                    'close': round(data_with_ha['HA_Close'].iloc[-1], 2),
                    'trend': 'Bullish' if data_with_ha['HA_Close'].iloc[-1] > data_with_ha['HA_Open'].iloc[-1] else 'Bearish'
                },
                'moving_averages': {
                    'ema_20': round(data_with_ma['EMA_20'].iloc[-1], 2) if not pd.isna(data_with_ma['EMA_20'].iloc[-1]) else None,
                    'sma_50': round(data_with_ma['SMA_50'].iloc[-1], 2) if not pd.isna(data_with_ma['SMA_50'].iloc[-1]) else None,
                    'sma_200': round(data_with_ma['SMA_200'].iloc[-1], 2) if not pd.isna(data_with_ma['SMA_200'].iloc[-1]) else None
                },
                'pivot_points': pivot_points,
                'volume_profile': volume_profile
            }
            
        except Exception as e:
            logger.error(f"Error calculating all indicators: {str(e)}")
            return {}