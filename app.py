import os

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import base64
import json
import time
import warnings
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

app = Flask(__name__)
CORS(app)

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
    
    if '.' not in symbol:
        symbol += '.NS'  # Default to NSE
    
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            
        if data.empty:
            return jsonify({"error": "No data found for symbol"}), 404
        
        # Calculate technical indicators
        data['RSI'] = ta.rsi(data['Close'])
        macd = ta.macd(data['Close'])
        data = pd.concat([data, macd], axis=1)
        data['EMA_20'] = ta.ema(data['Close'], length=20)
        data['EMA_50'] = ta.ema(data['Close'], length=50)
        data['EMA_200'] = ta.ema(data['Close'], length=200)
        data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
        data['VWAP'] = ta.vwap(data['High'], data['Low'], data['Close'], data['Volume'])
        
        # Prepare data for prediction
        data = data.dropna()
        if len(data) < 20:
            return jsonify({"error": "Insufficient data for analysis"}), 400
            
        data['Target'] = data['Close'].shift(-1)
        features = data[['Close', 'RSI', 'MACD_12_26_9', 'EMA_20', 'EMA_50', 'VWAP', 'ATR']].iloc[:-1]
        targets = data['Target'].iloc[:-1]
        
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
            ta_handler = TA_Handler(
                symbol=symbol,
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
        news_data = get_news_sentiment(symbol)
        
        # Generate charts
        chart_data = data.iloc[-100:] if len(data) > 100 else data
        chart_img = generate_chart(chart_data, symbol, interval)
        ichimoku_img = generate_ichimoku_chart(chart_data, symbol, interval) if advanced else None
        
        # Advanced analysis
        chart_patterns = detect_chart_patterns(chart_data)
        risk_metrics = calculate_risk_metrics(data)
        fib_levels = fibonacci_levels(chart_data)
        volume_profile = get_volume_profile(chart_data)
        trading_signals = generate_trading_signals(chart_data)
        
        # Prepare response
        response = {
            "symbol": symbol,
            "interval": interval,
            "current_price": data['Close'].iloc[-1],
            "predicted_price": float(prediction),
            "price_change": round(data['Close'].iloc[-1] - data['Open'].iloc[-1], 2),
            "technical_indicators": {
                "RSI": round(data['RSI'].iloc[-1], 2) if 'RSI' in data else None,
                "MACD": round(data['MACD_12_26_9'].iloc[-1], 4) if 'MACD_12_26_9' in data else None,
                "MACD_signal": round(data['MACDs_12_26_9'].iloc[-1], 4) if 'MACDs_12_26_9' in data else None,
                "EMA_20": round(data['EMA_20'].iloc[-1], 2) if 'EMA_20' in data else None,
                "EMA_50": round(data['EMA_50'].iloc[-1], 2) if 'EMA_50' in data else None,
                "EMA_200": round(data['EMA_200'].iloc[-1], 2) if 'EMA_200' in data else None,
                "VWAP": round(data['VWAP'].iloc[-1], 2) if 'VWAP' in data else None,
                "ATR": round(data['ATR'].iloc[-1], 2) if 'ATR' in data else None,
            },
            "tradingview_summary": tv_summary,
            "news_sentiment": news_data,
            "chart_image": chart_img,
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
        return jsonify({"error": str(e)}), 500

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # Disable debug mode for production
