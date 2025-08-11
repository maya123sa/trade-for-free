import os

# Suppress warnings at the very beginning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import base64
import json
import time
from io import BytesIO
from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import mplfinance as mpf
from tradingview_ta import TA_Handler, Interval
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from bs4 import BeautifulSoup
from scipy.stats import zscore
from scipy.signal import argrelextrema
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)
CORS(app)

# Enhanced symbol mapping for Indian stocks
INDIAN_STOCK_SYMBOLS = {
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'INFY': 'INFY.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'SBIN': 'SBIN.NS',
    'ITC': 'ITC.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'LT': 'LT.NS',
    'AXISBANK': 'AXISBANK.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'MARUTI': 'MARUTI.NS',
    'NESTLEIND': 'NESTLEIND.NS',
    'HCLTECH': 'HCLTECH.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'M&M': 'M&M.NS',
    'SUNPHARMA': 'SUNPHARMA.NS',
    'TITAN': 'TITAN.NS',
    'ULTRACEMCO': 'ULTRACEMCO.NS',
    'WIPRO': 'WIPRO.NS',
    'NTPC': 'NTPC.NS',
    'JSWSTEEL': 'JSWSTEEL.NS',
    'POWERGRID': 'POWERGRID.NS',
    'TATASTEEL': 'TATASTEEL.NS',
    'TECHM': 'TECHM.NS',
    'HINDALCO': 'HINDALCO.NS',
    'COALINDIA': 'COALINDIA.NS',
    'INDUSINDBK': 'INDUSINDBK.NS',
    'GRASIM': 'GRASIM.NS',
    'CIPLA': 'CIPLA.NS',
    'DRREDDY': 'DRREDDY.NS',
    'EICHERMOT': 'EICHERMOT.NS',
    'UPL': 'UPL.NS',
    'BAJAJ-AUTO': 'BAJAJ-AUTO.NS',
    'ONGC': 'ONGC.NS',
    'TATAMOTORS': 'TATAMOTORS.NS',
    'SBILIFE': 'SBILIFE.NS',
    'HDFCLIFE': 'HDFCLIFE.NS',
    'BRITANNIA': 'BRITANNIA.NS',
    'ADANIPORTS': 'ADANIPORTS.NS',
    'HEROMOTOCO': 'HEROMOTOCO.NS',
    'DIVISLAB': 'DIVISLAB.NS',
    'SHREECEM': 'SHREECEM.NS',
    'BAJAJFINSV': 'BAJAJFINSV.NS',
    'BPCL': 'BPCL.NS',
    'IOC': 'IOC.NS',
    'TATACONSUM': 'TATACONSUM.NS',
    # Indices
    'NIFTY': '^NSEI',
    'SENSEX': '^BSESN',
    'BANKNIFTY': '^NSEBANK',
    'NIFTYIT': '^CNXIT',
    'NIFTYPHARMA': '^CNXPHARMA'
}

# Indian timezone
IST = pytz.timezone('Asia/Kolkata')

def normalize_symbol(symbol):
    """Normalize symbol for Indian markets"""
    symbol = symbol.upper().strip()
    
    # If already has exchange suffix, return as is
    if symbol.endswith('.NS') or symbol.endswith('.BO') or symbol.startswith('^'):
        return symbol
    
    # Check our mapping first
    if symbol in INDIAN_STOCK_SYMBOLS:
        return INDIAN_STOCK_SYMBOLS[symbol]
    
    # Default to NSE
    return f"{symbol}.NS"

def get_stock_data_with_fallback(symbol, period='1y', interval='1d'):
    """Get stock data with multiple fallback options"""
    normalized_symbol = normalize_symbol(symbol)
    
    try:
        # Try primary symbol
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(normalized_symbol, period=period, interval=interval, progress=False)
            
        if not data.empty:
            return data, normalized_symbol
    except Exception as e:
        print(f"Primary fetch failed for {normalized_symbol}: {e}")
        pass
    
    # Try BSE if NSE failed
    if normalized_symbol.endswith('.NS'):
        bse_symbol = normalized_symbol.replace('.NS', '.BO')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(bse_symbol, period=period, interval=interval, progress=False)
                
            if not data.empty:
                return data, bse_symbol
        except Exception as e:
            print(f"BSE fetch failed for {bse_symbol}: {e}")
            pass
    
    # Try without exchange suffix
    base_symbol = symbol.replace('.NS', '').replace('.BO', '')
    for suffix in ['.NS', '.BO']:
        try:
            test_symbol = f"{base_symbol}{suffix}"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = yf.download(test_symbol, period=period, interval=interval, progress=False)
                
            if not data.empty:
                return data, test_symbol
        except Exception as e:
            print(f"Fallback fetch failed for {test_symbol}: {e}")
            continue
    
    # If all fails, return empty DataFrame
    return pd.DataFrame(), symbol

def calculate_advanced_indicators(data):
    """Calculate advanced technical indicators"""
    if len(data) < 20:
        return data
    
    try:
        # Stochastic Oscillator
        stoch = ta.stoch(data['High'], data['Low'], data['Close'])
        if stoch is not None and not stoch.empty:
            if 'STOCHk_14_3_3' in stoch.columns:
                data['%K'] = stoch['STOCHk_14_3_3']
            if 'STOCHd_14_3_3' in stoch.columns:
                data['%D'] = stoch['STOCHd_14_3_3']
        
        # Williams %R
        willr = ta.willr(data['High'], data['Low'], data['Close'])
        if willr is not None:
            data['Williams_%R'] = willr
        
        # Commodity Channel Index
        cci = ta.cci(data['High'], data['Low'], data['Close'])
        if cci is not None:
            data['CCI'] = cci
        
        # Money Flow Index
        mfi = ta.mfi(data['High'], data['Low'], data['Close'], data['Volume'])
        if mfi is not None:
            data['MFI'] = mfi
        
        # Average Directional Index
        adx = ta.adx(data['High'], data['Low'], data['Close'])
        if adx is not None and not adx.empty:
            if 'ADX_14' in adx.columns:
                data['ADX'] = adx['ADX_14']
        
        # Parabolic SAR
        psar = ta.psar(data['High'], data['Low'])
        if psar is not None and not psar.empty:
            if 'PSARl_0.02_0.2' in psar.columns:
                data['PSAR'] = psar['PSARl_0.02_0.2']
        
        # Ichimoku Cloud
        ichimoku = ta.ichimoku(data['High'], data['Low'], data['Close'])
        if ichimoku is not None and not ichimoku.empty:
            for col in ichimoku.columns:
                data[col] = ichimoku[col]
    except Exception as e:
        print(f"Error calculating advanced indicators: {e}")
    
    return data

def detect_candlestick_patterns(data):
    """Detect candlestick patterns"""
    patterns = {}
    
    if len(data) < 5:
        return patterns
    
    # Get last 5 candles
    recent = data.tail(5)
    
    # Doji pattern
    last_candle = recent.iloc[-1]
    body_size = abs(last_candle['Close'] - last_candle['Open'])
    candle_range = last_candle['High'] - last_candle['Low']
    
    if body_size < 0.1 * candle_range:
        patterns['doji'] = True
    
    # Hammer pattern
    lower_shadow = last_candle['Open'] - last_candle['Low'] if last_candle['Close'] > last_candle['Open'] else last_candle['Close'] - last_candle['Low']
    upper_shadow = last_candle['High'] - max(last_candle['Open'], last_candle['Close'])
    
    if lower_shadow > 2 * body_size and upper_shadow < 0.1 * candle_range:
        patterns['hammer'] = True
    
    # Shooting star
    if upper_shadow > 2 * body_size and lower_shadow < 0.1 * candle_range:
        patterns['shooting_star'] = True
    
    # Engulfing patterns
    if len(recent) >= 2:
        prev_candle = recent.iloc[-2]
        curr_candle = recent.iloc[-1]
        
        # Bullish engulfing
        if (prev_candle['Close'] < prev_candle['Open'] and  # Previous red
            curr_candle['Close'] > curr_candle['Open'] and  # Current green
            curr_candle['Open'] < prev_candle['Close'] and  # Opens below prev close
            curr_candle['Close'] > prev_candle['Open']):    # Closes above prev open
            patterns['bullish_engulfing'] = True
        
        # Bearish engulfing
        if (prev_candle['Close'] > prev_candle['Open'] and  # Previous green
            curr_candle['Close'] < curr_candle['Open'] and  # Current red
            curr_candle['Open'] > prev_candle['Close'] and  # Opens above prev close
            curr_candle['Close'] < prev_candle['Open']):    # Closes below prev open
            patterns['bearish_engulfing'] = True
    
    return patterns

def calculate_market_sentiment(data):
    """Calculate market sentiment indicators"""
    sentiment = {}
    
    if len(data) < 20:
        return sentiment
    
    # Fear & Greed components
    
    # 1. Price momentum (RSI)
    rsi = data['RSI'].iloc[-1] if 'RSI' in data else 50
    if rsi > 70:
        sentiment['rsi_sentiment'] = 'Greed'
    elif rsi < 30:
        sentiment['rsi_sentiment'] = 'Fear'
    else:
        sentiment['rsi_sentiment'] = 'Neutral'
    
    # 2. Price strength (% above 50-day MA)
    if 'EMA_50' in data:
        current_price = data['Close'].iloc[-1]
        ma_50 = data['EMA_50'].iloc[-1]
        if not pd.isna(ma_50):
            pct_above_ma = ((current_price - ma_50) / ma_50) * 100
            if pct_above_ma > 5:
                sentiment['ma_sentiment'] = 'Bullish'
            elif pct_above_ma < -5:
                sentiment['ma_sentiment'] = 'Bearish'
            else:
                sentiment['ma_sentiment'] = 'Neutral'
    
    # 3. Volume trend
    recent_volume = data['Volume'].tail(5).mean()
    avg_volume = data['Volume'].mean()
    volume_ratio = recent_volume / avg_volume
    
    if volume_ratio > 1.5:
        sentiment['volume_sentiment'] = 'High Interest'
    elif volume_ratio < 0.7:
        sentiment['volume_sentiment'] = 'Low Interest'
    else:
        sentiment['volume_sentiment'] = 'Normal'
    
    # 4. Volatility (ATR)
    if 'ATR' in data:
        current_atr = data['ATR'].iloc[-1]
        avg_atr = data['ATR'].mean()
        if not pd.isna(current_atr) and not pd.isna(avg_atr):
            if current_atr > 1.5 * avg_atr:
                sentiment['volatility'] = 'High'
            elif current_atr < 0.7 * avg_atr:
                sentiment['volatility'] = 'Low'
            else:
                sentiment['volatility'] = 'Normal'
    
    return sentiment

def get_intraday_levels(data):
    """Calculate intraday support/resistance levels"""
    levels = {}
    
    if len(data) < 10:
        return levels
    
    # Previous day high/low
    if len(data) >= 2:
        prev_high = data['High'].iloc[-2]
        prev_low = data['Low'].iloc[-2]
        prev_close = data['Close'].iloc[-2]
        
        levels['prev_high'] = round(prev_high, 2)
        levels['prev_low'] = round(prev_low, 2)
        levels['prev_close'] = round(prev_close, 2)
    
    # Opening price (for intraday)
    levels['open'] = round(data['Open'].iloc[-1], 2)
    
    # VWAP as dynamic support/resistance
    if 'VWAP' in data:
        levels['vwap'] = round(data['VWAP'].iloc[-1], 2)
    
    # Psychological levels (round numbers)
    current_price = data['Close'].iloc[-1]
    
    # Find nearest round numbers
    if current_price > 100:
        step = 50 if current_price < 1000 else 100
    else:
        step = 10
    
    lower_round = int(current_price // step) * step
    upper_round = lower_round + step
    
    levels['psychological_support'] = lower_round
    levels['psychological_resistance'] = upper_round
    
    return levels

def calculate_risk_reward_ratio(entry_price, stop_loss, target_price):
    """Calculate risk-reward ratio"""
    if not all([entry_price, stop_loss, target_price]):
        return None
    
    risk = abs(entry_price - stop_loss)
    reward = abs(target_price - entry_price)
    
    if risk == 0:
        return None
    
    return round(reward / risk, 2)

def get_market_hours_info():
    """Get current market hours information"""
    now = datetime.now(IST)
    current_time = now.time()
    
    # Market timings
    pre_open = datetime.strptime('09:00', '%H:%M').time()
    market_open = datetime.strptime('09:15', '%H:%M').time()
    market_close = datetime.strptime('15:30', '%H:%M').time()
    
    status = 'CLOSED'
    message = 'Market is closed'
    
    if pre_open <= current_time < market_open:
        status = 'PRE_OPEN'
        message = 'Pre-market session'
    elif market_open <= current_time < market_close:
        status = 'OPEN'
        message = 'Market is open'
    
    # Check if it's a weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        status = 'CLOSED'
        message = 'Weekend - Market closed'
    
    return {
        'status': status,
        'message': message,
        'current_time': now.strftime('%H:%M:%S IST'),
        'next_open': '09:15 IST' if status == 'CLOSED' else None
    }
# Add root route to prevent 404 errors
@app.route('/')
def home():
    return "Indian Trade Analyst API is running. Use /analyze?symbol=SYMBOL&interval=INTERVAL"

# Enhanced interval mapping
INTERVAL_MAP = {
    '1m': Interval.INTERVAL_1_MINUTE,
    '5m': Interval.INTERVAL_5_MINUTES,
    '15m': Interval.INTERVAL_15_MINUTES,
    '30m': Interval.INTERVAL_30_MINUTES,
    '1h': Interval.INTERVAL_1_HOUR,
    '4h': Interval.INTERVAL_4_HOURS,
    '1d': Interval.INTERVAL_1_DAY,
    '1w': Interval.INTERVAL_1_WEEK
}

# Cache setup
data_cache = {}
model_cache = {}
CACHE_EXPIRY = 300  # 5 minutes

def get_news_sentiment(symbol):
    """Fetch news and perform sentiment analysis - simplified to avoid transformers"""
    try:
        symbol = symbol.split('.')[0]  # Remove exchange suffix
        url = f"https://www.moneycontrol.com/rss/{symbol.lower()}_news.xml"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'xml')
        
        items = soup.find_all('item')[:5]  # Get top 5 news
        news_items = []
        positive_keywords = ['buy', 'strong', 'growth', 'positive', 'upgrade']
        negative_keywords = ['sell', 'weak', 'decline', 'negative', 'downgrade']
        
        for item in items:
            title = item.title.text
            description = item.description.text
            pub_date = item.pubDate.text
            
            # Simple sentiment analysis
            content = (title + " " + description).lower()
            positive_count = sum(keyword in content for keyword in positive_keywords)
            negative_count = sum(keyword in content for keyword in negative_keywords)
            
            sentiment = "NEUTRAL"
            if positive_count > negative_count:
                sentiment = "POSITIVE"
            elif negative_count > positive_count:
                sentiment = "NEGATIVE"
                
            news_items.append({
                "title": title,
                "description": description,
                "date": pub_date,
                "sentiment": sentiment
            })
        
        return {"news": news_items}
    except Exception as e:
        print(f"News sentiment error: {e}")
        return {"news": []}

def detect_chart_patterns(data):
    """Identify common chart patterns"""
    patterns = {}
    
    if len(data) < 20:
        return patterns  # Not enough data
    
    # Support/Resistance Levels
    highs = data['High']
    lows = data['Low']
    
    # Find local maxima/minima
    max_idx = argrelextrema(highs.values, np.greater, order=5)[0]
    min_idx = argrelextrema(lows.values, np.less, order=5)[0]
    
    support_levels = lows.iloc[min_idx].tail(3).values if len(min_idx) > 0 else []
    resistance_levels = highs.iloc[max_idx].tail(3).values if len(max_idx) > 0 else []
    
    patterns['support'] = support_levels.tolist()
    patterns['resistance'] = resistance_levels.tolist()
    
    # Detect Double Top/Bottom
    if len(resistance_levels) >= 2:
        if abs(resistance_levels[-1] - resistance_levels[-2]) < 0.02 * resistance_levels[-1]:
            patterns['double_top'] = True
    
    if len(support_levels) >= 2:
        if abs(support_levels[-1] - support_levels[-2]) < 0.02 * support_levels[-1]:
            patterns['double_bottom'] = True
    
    # Detect Head and Shoulders
    if len(max_idx) >= 4:
        head_idx = max_idx[-2]
        left_shoulder = highs.iloc[max_idx[-3]]
        head = highs.iloc[head_idx]
        right_shoulder = highs.iloc[max_idx[-1]]
        
        if (left_shoulder < head and right_shoulder < head and 
            abs(left_shoulder - right_shoulder) < 0.03 * head):
            patterns['head_shoulders'] = True
    
    return patterns

def calculate_risk_metrics(data):
    """Calculate advanced risk metrics"""
    if len(data) < 2:
        return {}
        
    returns = data['Close'].pct_change().dropna()
    
    metrics = {
        "volatility": returns.std() * np.sqrt(252),  # Annualized volatility
        "max_drawdown": (data['Close'] / data['Close'].cummax() - 1).min(),
    }
    
    # Only calculate if we have enough data
    if len(returns) > 1:
        metrics["sharpe_ratio"] = returns.mean() / returns.std() * np.sqrt(252)
        metrics["value_at_risk_95"] = np.percentile(returns, 5) * 100
    
    # Calculate correlations with Nifty 50
    try:
        nifty = yf.download('^NSEI', period='1y', interval='1d')['Close']
        aligned_data = data['Close'].reindex(nifty.index).dropna()
        aligned_nifty = nifty.reindex(aligned_data.index)
        
        if len(aligned_data) > 1 and len(aligned_nifty) > 1:
            corr = aligned_data.corr(aligned_nifty)
            metrics['nifty_correlation'] = corr
    except:
        pass
    
    return metrics

def fibonacci_levels(data):
    """Calculate Fibonacci retracement levels"""
    if len(data) < 5:
        return {}
        
    high = data['High'].max()
    low = data['Low'].min()
    diff = high - low
    
    return {
        "0%": high,
        "23.6%": high - diff * 0.236,
        "38.2%": high - diff * 0.382,
        "50%": high - diff * 0.5,
        "61.8%": high - diff * 0.618,
        "100%": low
    }

def generate_chart(data, symbol, interval):
    """Generate candlestick chart with technical indicators"""
    if len(data) < 20:
        return None
        
    # Create subplots
    apds = []
    
    # Only add indicators if they exist
    if 'EMA_20' in data:
        apds.append(mpf.make_addplot(data['EMA_20'], color='blue', panel=0))
    if 'EMA_50' in data:
        apds.append(mpf.make_addplot(data['EMA_50'], color='orange', panel=0))
    if 'RSI' in data:
        apds.append(mpf.make_addplot(data['RSI'], panel=1, color='purple', ylabel='RSI'))
        apds.append(mpf.make_addplot([70]*len(data), panel=1, color='red', linestyle='--'))
        apds.append(mpf.make_addplot([30]*len(data), panel=1, color='green', linestyle='--'))
    
    # Add MACD if available
    if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns:
        apds.append(mpf.make_addplot(data['MACD_12_26_9'], panel=2, color='blue', ylabel='MACD'))
        apds.append(mpf.make_addplot(data['MACDs_12_26_9'], panel=2, color='orange'))
        if 'MACDh_12_26_9' in data.columns:
            apds.append(mpf.make_addplot(data['MACDh_12_26_9'], type='bar', panel=2, color='gray', alpha=0.5))
    
    # Add VWAP if available
    if 'VWAP' in data.columns:
        apds.append(mpf.make_addplot(data['VWAP'], color='red', panel=0))
    
    # Add Bollinger Bands if available
    if 'BBU_20_2.0' in data.columns and 'BBL_20_2.0' in data.columns:
        apds.append(mpf.make_addplot(data['BBU_20_2.0'], color='teal', panel=0))
        apds.append(mpf.make_addplot(data['BBL_20_2.0'], color='teal', panel=0))
    
    img_bytes = BytesIO()
    
    try:
        mpf.plot(data, 
                type='candle', 
                style='charles',
                title=f'{symbol} - {interval}',
                addplot=apds,
                volume=True,
                figratio=(12,8),
                savefig=dict(fname=img_bytes, dpi=100, bbox_inches='tight'),
                panel_ratios=(6,2,2) if 'MACD_12_26_9' in data else (6,2),
                show_nontrading=False)
    except Exception as e:
        print(f"Chart generation error: {e}")
        return None
    
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode('utf-8')

def generate_ichimoku_chart(data, symbol, interval):
    """Generate chart with Ichimoku Cloud"""
    if len(data) < 52:
        return None
        
    # Calculate Ichimoku components
    tenkan = (data['High'].rolling(9).max() + data['Low'].rolling(9).min()) / 2
    kijun = (data['High'].rolling(26).max() + data['Low'].rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((data['High'].rolling(52).max() + data['Low'].rolling(52).min()) / 2).shift(26)
    
    # Create plot
    apds = [
        mpf.make_addplot(tenkan, color='blue', panel=0),
        mpf.make_addplot(kijun, color='red', panel=0),
        mpf.make_addplot(senkou_a, color='green', panel=0),
        mpf.make_addplot(senkou_b, color='orange', panel=0),
    ]
    
    img_bytes = BytesIO()
    
    try:
        fig, axes = mpf.plot(data, 
                type='candle', 
                style='charles',
                title=f'{symbol} Ichimoku - {interval}',
                addplot=apds,
                volume=True,
                figratio=(12,8),
                returnfig=True,
                panel_ratios=(6,2),
                show_nontrading=False)
        
        # Fill the cloud area
        ax = axes[0]
        ax.fill_between(data.index, senkou_a, senkou_b, 
                    where=senkou_a >= senkou_b, 
                    facecolor='lightgreen', alpha=0.3)
        ax.fill_between(data.index, senkou_a, senkou_b, 
                    where=senkou_a < senkou_b, 
                    facecolor='lightcoral', alpha=0.3)
        
        plt.savefig(img_bytes, format='png', dpi=100, bbox_inches='tight')
        img_bytes.seek(0)
        plt.close(fig)
    except Exception as e:
        print(f"Ichimoku chart error: {e}")
        return None
    
    return base64.b64encode(img_bytes.read()).decode('utf-8')

def get_volume_profile(data):
    """Calculate volume profile for significant price levels"""
    if len(data) < 10:
        return []
        
    bins = 20
    vp, edges = np.histogram(data['Close'], bins=bins, weights=data['Volume'])
    max_vol = vp.max()
    if max_vol > 0:
        significant_levels = edges[:-1][vp > 0.5 * max_vol]
        return significant_levels.tolist()
    return []

def generate_trading_signals(data):
    """Generate trading signals based on multiple indicators"""
    signals = []
    if len(data) < 20:
        return signals
    
    # MACD Crossover
    if 'MACD_12_26_9' in data and 'MACDs_12_26_9' in data:
        data['MACD_Cross'] = np.where(data['MACD_12_26_9'] > data['MACDs_12_26_9'], 1, -1)
        crossovers = data['MACD_Cross'].diff()
        if crossovers.iloc[-1] > 0:
            signals.append({"type": "MACD", "signal": "BUY", "strength": "Strong"})
        elif crossovers.iloc[-1] < 0:
            signals.append({"type": "MACD", "signal": "SELL", "strength": "Strong"})
    
    # RSI Divergence
    if 'RSI' in data and len(data) > 14:
        rsi = data['RSI']
        price = data['Close']
        
        # Look for bullish divergence
        if (price.iloc[-1] < price.iloc[-2] and 
            rsi.iloc[-1] > rsi.iloc[-2] and 
            rsi.iloc[-1] < 40):
            signals.append({"type": "RSI", "signal": "BUY", "strength": "Medium", "reason": "Bullish Divergence"})
        
        # Look for bearish divergence
        elif (price.iloc[-1] > price.iloc[-2] and 
              rsi.iloc[-1] < rsi.iloc[-2] and 
              rsi.iloc[-1] > 60):
            signals.append({"type": "RSI", "signal": "SELL", "strength": "Medium", "reason": "Bearish Divergence"})
    
    # Volume Spike
    if 'Volume' in data:
        avg_volume = data['Volume'].mean()
        if data['Volume'].iloc[-1] > 2 * avg_volume:
            if data['Close'].iloc[-1] > data['Open'].iloc[-1]:
                signals.append({"type": "VOLUME", "signal": "BUY", "strength": "Weak", "reason": "Bullish Volume Spike"})
            else:
                signals.append({"type": "VOLUME", "signal": "SELL", "strength": "Weak", "reason": "Bearish Volume Spike"})
    
    # Golden/Death Cross
    if 'EMA_50' in data and 'EMA_200' in data:
        if data['EMA_50'].iloc[-1] > data['EMA_200'].iloc[-1] and data['EMA_50'].iloc[-2] <= data['EMA_200'].iloc[-2]:
            signals.append({"type": "MOVING_AVG", "signal": "BUY", "strength": "Strong", "reason": "Golden Cross"})
        elif data['EMA_50'].iloc[-1] < data['EMA_200'].iloc[-1] and data['EMA_50'].iloc[-2] >= data['EMA_200'].iloc[-2]:
            signals.append({"type": "MOVING_AVG", "signal": "SELL", "strength": "Strong", "reason": "Death Cross"})
    
    # Bollinger Band Squeeze
    if 'BBU_20_2.0' in data and 'BBL_20_2.0' in data and 'BBM_20_2.0' in data:
        bb_width = (data['BBU_20_2.0'] - data['BBL_20_2.0']) / data['BBM_20_2.0']
        if bb_width.iloc[-1] < 0.1:
            signals.append({"type": "BOLLINGER", "signal": "WATCH", "strength": "Medium", "reason": "Bollinger Squeeze"})
    
    return signals

@app.route('/analyze', methods=['GET'])
def analyze():
    symbol = request.args.get('symbol', 'RELIANCE.NS').upper()
    interval = request.args.get('interval', '1d')
    advanced = request.args.get('advanced', 'false').lower() == 'true'
    
    # Normalize symbol
    original_symbol = symbol
    symbol = normalize_symbol(symbol)
    
    # Check cache
    cache_key = f"{symbol}_{interval}"
    if cache_key in data_cache and time.time() - data_cache[cache_key]['timestamp'] < CACHE_EXPIRY:
        return jsonify(data_cache[cache_key]['response'])
    
    try:
        # Determine period based on interval
        period_map = {
            '1m': '7d',
            '5m': '30d',
            '15m': '60d',
            '30m': '120d',
            '1h': '200d',
            '4h': '1y',
            '1d': '5y',
            '1w': '10y'
        }
        period = period_map.get(interval, '1y')
        
        # Fetch historical data
        data, actual_symbol = get_stock_data_with_fallback(original_symbol, period=period, interval=interval)
            
        if data.empty:
            return jsonify({"error": f"No data found for symbol {original_symbol}. Please check the symbol name."}), 404
        
        # Calculate technical indicators
        data['RSI'] = ta.rsi(data['Close'])
        macd = ta.macd(data['Close'])
        if macd is not None and not macd.empty:
            data = pd.concat([data, macd], axis=1)
        data['EMA_20'] = ta.ema(data['Close'], length=20)
        data['EMA_50'] = ta.ema(data['Close'], length=50)
        data['EMA_200'] = ta.ema(data['Close'], length=200)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
        
        # Add Bollinger Bands
        bb = ta.bbands(data['Close'])
        if bb is not None and not bb.empty:
            data = pd.concat([data, bb], axis=1)
        
        # Calculate advanced indicators
        data = calculate_advanced_indicators(data)
        
        # Prepare data for prediction
        data = data.dropna()
        if len(data) < 20:
            return jsonify({"error": "Insufficient data for analysis"}), 400
            
        data['Target'] = data['Close'].shift(-1)
        
        # Select available features
        feature_columns = ['Close']
        for col in ['RSI', 'MACD_12_26_9', 'EMA_20', 'EMA_50', 'VWAP', 'ATR']:
            if col in data.columns and not data[col].isna().all():
                feature_columns.append(col)
        
        features = data[feature_columns].iloc[:-1]
        targets = data['Target'].iloc[:-1]
        
        # Remove rows with NaN values
        valid_indices = features.dropna().index
        features = features.loc[valid_indices]
        targets = targets.loc[valid_indices]
        
        if len(features) < 10:
            return jsonify({"error": "Insufficient clean data for prediction"}), 400
        
        # Train prediction model (with caching)
        model_key = f"model_{interval}"
        if model_key not in model_cache or time.time() - model_cache[model_key]['timestamp'] > 86400:
            # Use simpler model if data is small
            n_estimators = 100 if len(features) < 500 else 500
            
            # Handle case where we might have too few samples
            test_size = 0.2 if len(features) > 100 else 0.1
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=test_size, shuffle=False
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            model = XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, max_depth=6)
            model.fit(X_train_scaled, y_train)
            
            # Save to cache
            model_cache[model_key] = {
                'model': model,
                'scaler': scaler,
                'timestamp': time.time()
            }
        else:
            model = model_cache[model_key]['model']
            scaler = model_cache[model_key]['scaler']
        
        # Make prediction
        last_data = features.iloc[-1].values.reshape(1, -1)
        last_data_scaled = scaler.transform(last_data)
        prediction = model.predict(last_data_scaled)[0]
        
        # Get TradingView analysis
        tv_interval = INTERVAL_MAP.get(interval, Interval.INTERVAL_1_DAY)
        try:
            # Clean symbol for TradingView
            tv_symbol = original_symbol.replace('.NS', '').replace('.BO', '')
            ta_handler = TA_Handler(
                symbol=tv_symbol,
                screener="india",
                exchange="NSE",
                interval=tv_interval
            )
            ta_analysis = ta_handler.get_analysis()
            tv_summary = {
                "recommendation": ta_analysis.summary['RECOMMENDATION'],
                "buy": ta_analysis.summary['BUY'],
                "sell": ta_analysis.summary['SELL'],
                "neutral": ta_analysis.summary['NEUTRAL'],
            }
        except Exception as e:
            print(f"TradingView error: {e}")
            tv_summary = {
                "recommendation": "ERROR",
                "buy": 0,
                "sell": 0,
                "neutral": 0
            }
        
        # Get news sentiment
        news_data = get_news_sentiment(original_symbol)
        
        # Generate charts
        chart_data = data.iloc[-100:] if len(data) > 100 else data
        chart_img = generate_chart(chart_data, original_symbol, interval)
        ichimoku_img = generate_ichimoku_chart(chart_data, original_symbol, interval) if advanced else None
        
        # Advanced analysis
        chart_patterns = detect_chart_patterns(chart_data)
        risk_metrics = calculate_risk_metrics(data)
        fib_levels = fibonacci_levels(chart_data)
        volume_profile = get_volume_profile(chart_data)
        trading_signals = generate_trading_signals(chart_data)
        
        # New advanced features
        candlestick_patterns = detect_candlestick_patterns(chart_data)
        market_sentiment = calculate_market_sentiment(data)
        intraday_levels = get_intraday_levels(chart_data)
        market_hours = get_market_hours_info()
        
        # Prepare response
        response = {
            "symbol": original_symbol,
            "actual_symbol": actual_symbol,
            "interval": interval,
            "current_price": data['Close'].iloc[-1],
            "predicted_price": float(prediction),
            "price_change": round(data['Close'].iloc[-1] - data['Open'].iloc[-1], 2),
            "price_change_pct": round(((data['Close'].iloc[-1] - data['Open'].iloc[-1]) / data['Open'].iloc[-1]) * 100, 2),
            "market_hours": market_hours,
            "technical_indicators": {
                "RSI": round(data['RSI'].iloc[-1], 2) if 'RSI' in data else None,
                "MACD": round(data['MACD_12_26_9'].iloc[-1], 4) if 'MACD_12_26_9' in data else None,
                "MACD_signal": round(data['MACDs_12_26_9'].iloc[-1], 4) if 'MACDs_12_26_9' in data else None,
                "EMA_20": round(data['EMA_20'].iloc[-1], 2) if 'EMA_20' in data else None,
                "EMA_50": round(data['EMA_50'].iloc[-1], 2) if 'EMA_50' in data else None,
                "EMA_200": round(data['EMA_200'].iloc[-1], 2) if 'EMA_200' in data else None,
                "VWAP": round(data['VWAP'].iloc[-1], 2) if 'VWAP' in data else None,
                "ATR": round(data['ATR'].iloc[-1], 2) if 'ATR' in data else None,
                "Stoch_K": round(data['%K'].iloc[-1], 2) if '%K' in data and not pd.isna(data['%K'].iloc[-1]) else None,
                "Stoch_D": round(data['%D'].iloc[-1], 2) if '%D' in data and not pd.isna(data['%D'].iloc[-1]) else None,
                "Williams_R": round(data['Williams_%R'].iloc[-1], 2) if 'Williams_%R' in data and not pd.isna(data['Williams_%R'].iloc[-1]) else None,
                "CCI": round(data['CCI'].iloc[-1], 2) if 'CCI' in data and not pd.isna(data['CCI'].iloc[-1]) else None,
                "MFI": round(data['MFI'].iloc[-1], 2) if 'MFI' in data and not pd.isna(data['MFI'].iloc[-1]) else None,
                "ADX": round(data['ADX'].iloc[-1], 2) if 'ADX' in data and not pd.isna(data['ADX'].iloc[-1]) else None,
                "PSAR": round(data['PSAR'].iloc[-1], 2) if 'PSAR' in data and not pd.isna(data['PSAR'].iloc[-1]) else None,
                "BB_Upper": round(data['BBU_20_2.0'].iloc[-1], 2) if 'BBU_20_2.0' in data.columns and not pd.isna(data['BBU_20_2.0'].iloc[-1]) else None,
                "BB_Middle": round(data['BBM_20_2.0'].iloc[-1], 2) if 'BBM_20_2.0' in data.columns and not pd.isna(data['BBM_20_2.0'].iloc[-1]) else None,
                "BB_Lower": round(data['BBL_20_2.0'].iloc[-1], 2) if 'BBL_20_2.0' in data.columns and not pd.isna(data['BBL_20_2.0'].iloc[-1]) else None,
            },
            "tradingview_summary": tv_summary,
            "news_sentiment": news_data,
            "chart_image": chart_img,
            "candlestick_patterns": candlestick_patterns,
            "market_sentiment": market_sentiment,
            "intraday_levels": intraday_levels,
            "advanced_analysis": {
                "ichimoku_chart": ichimoku_img,
                "chart_patterns": chart_patterns,
                "risk_metrics": risk_metrics,
                "fibonacci_levels": fib_levels,
                "volume_profile": volume_profile,
                "trading_signals": trading_signals
            } if advanced else None
        }
        
        # Cache response
        data_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error analyzing {original_symbol}: {str(e)}")
        return jsonify({"error": f"Error analyzing {original_symbol}: {str(e)}"}), 500

@app.route('/backtest', methods=['GET'])
def backtest_strategy():
    symbol = request.args.get('symbol', 'RELIANCE.NS').upper()
    strategy = request.args.get('strategy', 'rsi_macd')
    
    if '.' not in symbol:
        symbol += '.NS'
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(symbol, period='5y', interval='1d', progress=False)
            
        if data.empty:
            return jsonify({"error": "No data found for symbol"}), 404
        
        # Calculate indicators
        data['RSI'] = ta.rsi(data['Close'])
        macd = ta.macd(data['Close'])
        data = pd.concat([data, macd], axis=1)
        data['EMA_50'] = ta.ema(data['Close'], length=50)
        data['EMA_200'] = ta.ema(data['Close'], length=200)
        
        # Strategy logic
        if strategy == 'rsi_macd':
            data['Signal'] = np.where(
                (data['RSI'] < 30) & (data['MACD_12_26_9'] > data['MACDs_12_26_9']), 
                1,  # Buy
                np.where(
                    (data['RSI'] > 70) & (data['MACD_12_26_9'] < data['MACDs_12_26_9']), 
                    -1,  # Sell
                    0    # Hold
                )
            )
        elif strategy == 'golden_cross':
            data['Signal'] = np.where(
                (data['EMA_50'] > data['EMA_200']) & (data['EMA_50'].shift(1) <= data['EMA_200'].shift(1)), 
                1,  # Buy
                np.where(
                    (data['EMA_50'] < data['EMA_200']) & (data['EMA_50'].shift(1) >= data['EMA_200'].shift(1)), 
                    -1,  # Sell
                    0    # Hold
                )
            )
        
        # Calculate positions
        data['Position'] = data['Signal'].diff()
        
        # Calculate returns
        data['Daily_Return'] = data['Close'].pct_change()
        data['Strategy_Return'] = data['Daily_Return'] * data['Signal'].shift(1)
        
        # Cumulative returns
        data['Cumulative_Market'] = (1 + data['Daily_Return']).cumprod()
        data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()
        
        # Performance metrics
        total_return = data['Cumulative_Strategy'].iloc[-1] - 1
        max_drawdown = (data['Cumulative_Strategy'] / data['Cumulative_Strategy'].cummax() - 1).min()
        if len(data[data['Strategy_Return'].notnull()]) > 1:
            sharpe_ratio = data['Strategy_Return'].mean() / data['Strategy_Return'].std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        # Prepare response
        return jsonify({
            "symbol": symbol,
            "strategy": strategy,
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe_ratio),
            "trades": int(data['Position'].abs().sum() // 2),
            "final_value": float(100000 * data['Cumulative_Strategy'].iloc[-1])  # Starting with ₹100,000
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/screener', methods=['GET'])
def stock_screener():
    """Screen stocks based on technical criteria"""
    try:
        # Get screening criteria
        min_rsi = float(request.args.get('min_rsi', 30))
        max_rsi = float(request.args.get('max_rsi', 70))
        min_volume_ratio = float(request.args.get('min_volume_ratio', 1.5))
        price_above_ema = request.args.get('price_above_ema', 'false').lower() == 'true'
        
        # List of stocks to screen (Nifty 50)
        stocks_to_screen = [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK',
            'SBIN', 'ITC', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT',
            'MARUTI', 'NESTLEIND', 'HCLTECH', 'BAJFINANCE', 'M&M', 'SUNPHARMA',
            'TITAN', 'ULTRACEMCO', 'WIPRO', 'NTPC', 'JSWSTEEL', 'POWERGRID',
            'TATASTEEL', 'TECHM', 'HINDALCO', 'COALINDIA', 'TATAMOTORS'
        ]
        
        screened_stocks = []
        
        for stock in stocks_to_screen:
            try:
                data, actual_symbol = get_stock_data_with_fallback(stock, period='3mo', interval='1d')
                
                if data.empty or len(data) < 20:
                    continue
                
                # Calculate indicators
                data['RSI'] = ta.rsi(data['Close'])
                data['EMA_20'] = ta.ema(data['Close'], length=20)
                
                # Get latest values
                current_rsi = data['RSI'].iloc[-1]
                current_price = data['Close'].iloc[-1]
                current_ema = data['EMA_20'].iloc[-1]
                current_volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].mean()
                
                # Skip if any value is NaN
                if pd.isna(current_rsi) or pd.isna(current_ema) or pd.isna(current_volume):
                    continue
                
                volume_ratio = current_volume / avg_volume
                
                # Apply screening criteria
                meets_criteria = True
                
                if not (min_rsi <= current_rsi <= max_rsi):
                    meets_criteria = False
                
                if volume_ratio < min_volume_ratio:
                    meets_criteria = False
                
                if price_above_ema and current_price <= current_ema:
                    meets_criteria = False
                
                if meets_criteria:
                    if len(data) >= 2:
                        price_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
                    else:
                        price_change = 0
                    
                    screened_stocks.append({
                        'symbol': stock,
                        'price': round(current_price, 2),
                        'rsi': round(current_rsi, 2),
                        'volume_ratio': round(volume_ratio, 2),
                        'price_change_pct': round(price_change, 2),
                        'above_ema': current_price > current_ema
                    })
                    
            except Exception as e:
                print(f"Error screening {stock}: {str(e)}")
                continue
        
        # Sort by RSI
        screened_stocks.sort(key=lambda x: x['rsi'])
        
        return jsonify({
            'screened_stocks': screened_stocks,
            'total_found': len(screened_stocks),
            'criteria': {
                'min_rsi': min_rsi,
                'max_rsi': max_rsi,
                'min_volume_ratio': min_volume_ratio,
                'price_above_ema': price_above_ema
            },
            'timestamp': datetime.now(IST).isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/momentum', methods=['GET'])
def momentum_analysis():
    """Analyze momentum across different timeframes"""
    symbol = request.args.get('symbol', 'RELIANCE').upper()
    
    try:
        # Get data for different timeframes
        timeframes = {
            '1d': {'period': '1mo', 'interval': '1d'},
            '1w': {'period': '6mo', 'interval': '1wk'},
            '1mo': {'period': '2y', 'interval': '1mo'}
        }
        
        momentum_data = {}
        
        for tf, params in timeframes.items():
            data, _ = get_stock_data_with_fallback(symbol, period=params['period'], interval=params['interval'])
            
            if data.empty or len(data) < 10:
                continue
            
            # Calculate momentum indicators
            data['RSI'] = ta.rsi(data['Close'])
            data['ROC'] = ta.roc(data['Close'], length=10)  # Rate of Change
            
            # Price momentum
            current_price = data['Close'].iloc[-1]
            price_10_ago = data['Close'].iloc[-10] if len(data) >= 10 else data['Close'].iloc[0]
            momentum_pct = ((current_price - price_10_ago) / price_10_ago) * 100
            
            momentum_data[tf] = {
                'rsi': round(data['RSI'].iloc[-1], 2) if not pd.isna(data['RSI'].iloc[-1]) else None,
                'roc': round(data['ROC'].iloc[-1], 2) if not pd.isna(data['ROC'].iloc[-1]) else None,
                'momentum_pct': round(momentum_pct, 2),
                'trend': 'Bullish' if momentum_pct > 0 else 'Bearish'
            }
        
        # Overall momentum score
        scores = []
        for tf_data in momentum_data.values():
            if tf_data['rsi']:
                if tf_data['rsi'] > 50:
                    scores.append(1)
                else:
                    scores.append(-1)
            
            if tf_data['momentum_pct'] > 0:
                scores.append(1)
            else:
                scores.append(-1)
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        if overall_score > 0.3:
            overall_momentum = 'Strong Bullish'
        elif overall_score > 0:
            overall_momentum = 'Bullish'
        elif overall_score > -0.3:
            overall_momentum = 'Neutral'
        else:
            overall_momentum = 'Bearish'
        
        return jsonify({
            'symbol': symbol,
            'timeframes': momentum_data,
            'overall_momentum': overall_momentum,
            'momentum_score': round(overall_score, 2),
            'timestamp': datetime.now(IST).isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/alerts', methods=['GET'])
def price_alerts():
    """Get price alert suggestions based on technical levels"""
    symbol = request.args.get('symbol', 'RELIANCE').upper()
    
    try:
        data, actual_symbol = get_stock_data_with_fallback(symbol, period='3mo', interval='1d')
        
        if data.empty:
            return jsonify({"error": f"No data found for symbol {symbol}"}), 404
        
        # Calculate indicators
        data['RSI'] = ta.rsi(data['Close'])
        data['EMA_20'] = ta.ema(data['Close'], length=20)
        data['EMA_50'] = ta.ema(data['Close'], length=50)
        bb = ta.bbands(data['Close'])
        if bb is not None:
            data = pd.concat([data, bb], axis=1)
        
        current_price = data['Close'].iloc[-1]
        
        # Generate alert levels
        alerts = []
        
        # Support/Resistance levels
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        
        if current_price < recent_high * 0.98:
            alerts.append({
                'type': 'Resistance',
                'level': round(recent_high, 2),
                'distance_pct': round(((recent_high - current_price) / current_price) * 100, 2),
                'message': f'Watch for breakout above {recent_high}'
            })
        
        if current_price > recent_low * 1.02:
            alerts.append({
                'type': 'Support',
                'level': round(recent_low, 2),
                'distance_pct': round(((current_price - recent_low) / current_price) * 100, 2),
                'message': f'Watch for breakdown below {recent_low}'
            })
        
        # Moving average alerts
        if not pd.isna(data['EMA_20'].iloc[-1]):
            ema_20 = data['EMA_20'].iloc[-1]
            if abs(current_price - ema_20) / current_price < 0.02:  # Within 2%
                alerts.append({
                    'type': 'EMA_20',
                    'level': round(ema_20, 2),
                    'distance_pct': round(((ema_20 - current_price) / current_price) * 100, 2),
                    'message': f'Price near EMA 20 at {ema_20}'
                })
        
        # Bollinger Band alerts
        if 'BBU_20_2.0' in data and not pd.isna(data['BBU_20_2.0'].iloc[-1]):
            bb_upper = data['BBU_20_2.0'].iloc[-1]
            bb_lower = data['BBL_20_2.0'].iloc[-1]
            
            if current_price > bb_upper * 0.99:
                alerts.append({
                    'type': 'Bollinger_Upper',
                    'level': round(bb_upper, 2),
                    'distance_pct': round(((bb_upper - current_price) / current_price) * 100, 2),
                    'message': f'Price near upper Bollinger Band at {bb_upper}'
                })
            
            if current_price < bb_lower * 1.01:
                alerts.append({
                    'type': 'Bollinger_Lower',
                    'level': round(bb_lower, 2),
                    'distance_pct': round(((current_price - bb_lower) / current_price) * 100, 2),
                    'message': f'Price near lower Bollinger Band at {bb_lower}'
                })
        
        # RSI alerts
        if not pd.isna(data['RSI'].iloc[-1]):
            rsi = data['RSI'].iloc[-1]
            if rsi > 70:
                alerts.append({
                    'type': 'RSI_Overbought',
                    'level': round(rsi, 2),
                    'distance_pct': 0,
                    'message': f'RSI overbought at {rsi:.1f}'
                })
            elif rsi < 30:
                alerts.append({
                    'type': 'RSI_Oversold',
                    'level': round(rsi, 2),
                    'distance_pct': 0,
                    'message': f'RSI oversold at {rsi:.1f}'
                })
        
        return jsonify({
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'alerts': alerts,
            'total_alerts': len(alerts),
            'timestamp': datetime.now(IST).isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Disable debug mode for production