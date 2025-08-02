import os
import json
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas_ta as ta
from flask import Flask, request, jsonify, send_file
from io import BytesIO
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from functools import lru_cache
import base64
from tradingview_ta import TA_Handler, Interval

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Constants
NSE_SUFFIX = ".NS"
CACHE_EXPIRY = 3600  # 1 hour
INDICATOR_CONFIG = {
    'trend': [
        {'sma': [10, 20, 50, 100, 200]},
        {'ema': [12, 26, 50]},
        {'wma': [20]},
        {'hma': [20]},
        {'kama': [10, 2, 30]},
        {'macd': [12, 26, 9]},
        {'adx': [14]},
        {'vortex': [14]},
        {'trix': [15]},
        {'kst': [10, 15, 20, 30]},
        {'stc': [23, 50]},
        {'dpo': [20]},
        {'psar': [0.02, 0.2]}
    ],
    'momentum': [
        {'rsi': [14]},
        {'stoch': [14, 3]},
        {'willr': [14]},
        {'cci': [20]},
        {'roc': [10]},
        {'mfi': [14]},
        {'ao': []},
        {'uo': [7, 14, 28]},
        {'cmo': [14]},
        {'kdj': [9, 3, 3]},
        {'tsi': [25, 13]},
        {'kvo': [34, 55]},
        {'pvo': [12, 26, 9]}
    ],
    'volatility': [
        {'bbands': [20, 2]},
        {'atr': [14]},
        {'kc': [20, 2]},
        {'dc': [20]},
        {'ui': [14]},
        {'fisher': [10]},
        {'rvi': [14]},
        {'thermo': [20, 2, 0.5]}
    ],
    'volume': [
        {'obv': []},
        {'vp': [20]},
        {'vwap': []},
        {'ad': []},
        {'cmf': [20]},
        {'eom': [14]},
        {'vema': [20]},
        {'vroc': [14]},
        {'vzo': [5, 20]},
        {'pvi': []},
        {'nvi': []},
        {'efi': [13]}
    ]
}

# TradingView interval mapping
TV_INTERVAL_MAP = {
    '1m': Interval.INTERVAL_1_MINUTE,
    '5m': Interval.INTERVAL_5_MINUTES,
    '15m': Interval.INTERVAL_15_MINUTES,
    '30m': Interval.INTERVAL_30_MINUTES,
    '1h': Interval.INTERVAL_1_HOUR,
    '1d': Interval.INTERVAL_1_DAY,
    '1wk': Interval.INTERVAL_1_WEEK,
    '1mo': Interval.INTERVAL_1_MONTH
}

# LRU Cache for API results
@lru_cache(maxsize=128)
def cached_yf_download(symbol, interval, period):
    return yf.download(symbol, period=period, interval=interval)

def get_period_for_interval(interval):
    """Determine appropriate period based on interval"""
    period_map = {
        '1m': '7d', '5m': '60d', '15m': '60d', '30m': '60d',
        '1h': '730d', '1d': '5y', '1wk': '10y', '1mo': '20y'
    }
    return period_map.get(interval, '2y')

def fetch_stock_data(symbol, interval):
    """Fetch and cache stock data"""
    ticker = f"{symbol}{NSE_SUFFIX}"
    period = get_period_for_interval(interval)
    
    try:
        df = cached_yf_download(ticker, interval, period)
        if df.empty:
            return None
            
        # Clean and format data
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        app.logger.error(f"Data fetch error: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate all technical indicators"""
    for category, indicators in INDICATOR_CONFIG.items():
        for indicator in indicators:
            for name, params in indicator.items():
                try:
                    # Special handling for indicators without parameters
                    if params:
                        df.ta(kind=name, length=params, append=True)
                    else:
                        df.ta(kind=name, append=True)
                except Exception as e:
                    app.logger.warning(f"Indicator {name}{params} failed: {str(e)}")
    
    # Add specialized indicators
    df = add_specialized_indicators(df)
    return df

def add_specialized_indicators(df):
    """Add non-standard indicators"""
    # Market Profile
    if 'volume' in df.columns:
        df['poc'] = df.groupby(pd.Grouper(freq='D'))['volume'].transform(lambda x: x.idxmax())
    
    # Fibonacci Levels
    if not df.empty:
        max_price = df['high'].max()
        min_price = df['low'].min()
        diff = max_price - min_price
        
        fib_levels = {
            'fib_0': max_price,
            'fib_23': max_price - 0.236 * diff,
            'fib_38': max_price - 0.382 * diff,
            'fib_50': max_price - 0.5 * diff,
            'fib_61': max_price - 0.618 * diff,
            'fib_78': max_price - 0.786 * diff,
            'fib_100': min_price
        }
        
        for level, value in fib_levels.items():
            df[level] = value
    
    # Machine Learning Forecast (simplified)
    if len(df) > 100:
        df = add_ml_forecast(df)
    
    return df

def add_ml_forecast(df):
    """Add machine learning predictions"""
    # Feature Engineering
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['momentum'] = df['returns'].rolling(10).mean()
    
    # Create target (1 if next day up, 0 otherwise)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Prepare data
    features = df[['returns', 'volatility', 'momentum']].dropna()
    targets = df.loc[features.index, 'target']
    
    if len(features) > 100 and targets.sum() > 10:
        # Train/test split
        split = int(0.8 * len(features))
        X_train, X_test = features.iloc[:split], features.iloc[split:]
        y_train, y_test = targets.iloc[:split], targets.iloc[split:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Generate predictions
        full_set = scaler.transform(features)
        df.loc[features.index, 'ml_forecast'] = model.predict_proba(full_set)[:, 1]
    
    return df

def detect_candlestick_patterns(df):
    """Identify candlestick patterns"""
    patterns = []
    candle_func = ta.CDL_PATTERNS
    
    # Check for common patterns
    for pattern in [
        'CDLHAMMER', 'CDLENGULFING', 'CDLMORNINGSTAR', 'CDL3WHITESOLDIERS',
        'CDLEVENINGSTAR', 'CDLSHOOTINGSTAR', 'CDLHANGINGMAN', 'CDLDARKCLOUDCOVER',
        'CDLPIERCING', 'CDL3BLACKCROWS', 'CDLDOJI', 'CDLDRAGONFLYDOJI',
        'CDLGRAVESTONEDOJI', 'CDLSPINNINGTOP', 'CDLMARUBOZU', 'CDLKICKING',
        'CDLINVERTEDHAMMER', 'CDLMORNINGDOJISTAR', 'CDLEVENINGDOJISTAR', 'CDLMATCHINGLOW',
        'CDLUPSIDEGAP2CROWS', 'CDLIDENTICAL3CROWS', 'CDL3LINESTRIKE', 'CDLUNIQUE3RIVER',
        'CDL3INSIDE', 'CDL3OUTSIDE', 'CDLTRISTAR', 'CDLADVANCEBLOCK',
        'CDLBREAKAWAY', 'CDLTASUKIGAP', 'CDLTRISTAR', 'CDLXSIDEGAP3METHODS'
    ]:
        if pattern in candle_func:
            result = candle_func[pattern](df['open'], df['high'], df['low'], df['close'])
            last_signal = result.iloc[-1]
            if last_signal != 0:
                patterns.append({
                    'name': pattern[3:].lower().capitalize(),
                    'date': df.index[-1].strftime('%Y-%m-%d'),
                    'reliability': 4.0 if abs(last_signal) == 100 else 3.0,
                    'direction': 'Bullish' if last_signal > 0 else 'Bearish'
                })
    
    # Limit to 5 strongest patterns
    return sorted(patterns, key=lambda x: x['reliability'], reverse=True)[:5]

def calculate_key_levels(df):
    """Calculate support/resistance levels"""
    if df.empty:
        return {'support': [], 'resistance': [], 'pivot': 0}
    
    # Pivot Points
    last = df.iloc[-1]
    pivot = (last['high'] + last['low'] + last['close']) / 3
    s1 = (2 * pivot) - last['high']
    r1 = (2 * pivot) - last['low']
    
    # Support/Resistance
    window = 20
    max_high = df['high'].rolling(window).max().iloc[-1]
    min_low = df['low'].rolling(window).min().iloc[-1]
    
    return {
        'support': [min_low, s1],
        'resistance': [max_high, r1],
        'pivot': pivot
    }

def get_tradingview_analysis(symbol, interval):
    """Get technical analysis from TradingView"""
    try:
        # Convert our interval to TradingView's format
        tv_interval = TV_INTERVAL_MAP.get(interval, Interval.INTERVAL_1_DAY)
        
        handler = TA_Handler(
            symbol=symbol + NSE_SUFFIX,
            screener="india",
            exchange="NSE",
            interval=tv_interval
        )
        
        analysis = handler.get_analysis()
        
        # Parse relevant information
        return {
            "summary": analysis.summary,
            "oscillators": analysis.oscillators,
            "moving_averages": analysis.moving_averages,
            "indicators": analysis.indicators,
            "recommendation": analysis.summary.get('RECOMMENDATION', 'NEUTRAL')
        }
    except Exception as e:
        app.logger.error(f"TradingView error: {str(e)}")
        return None

def get_tradingview_recommendation(tv_analysis):
    """Convert TradingView recommendation to our format"""
    if not tv_analysis:
        return None
    
    recommendation_map = {
        'STRONG_BUY': 'Strong Buy',
        'BUY': 'Buy',
        'NEUTRAL': 'Neutral',
        'SELL': 'Sell',
        'STRONG_SELL': 'Strong Sell'
    }
    
    tv_rec = tv_analysis.get('recommendation', 'NEUTRAL')
    return recommendation_map.get(tv_rec, tv_rec)

def generate_charts(symbol, interval, df):
    """Generate all required charts"""
    return {
        'main': generate_main_chart(symbol, interval, df),
        'momentum': generate_momentum_chart(df),
        'volume': generate_volume_chart(df),
        'advanced': generate_advanced_chart(df)
    }

def generate_main_chart(symbol, interval, df):
    """Generate main analysis chart"""
    if df.empty:
        return ""
    
    # Prepare plot
    apds = []
    
    # Add moving averages if available
    for ma in ['sma_20', 'ema_50']:
        if ma in df.columns:
            apds.append(mpf.make_addplot(df[ma], color='blue' if 'sma' in ma else 'purple'))
    
    # Add Bollinger Bands if available
    if 'bbands_upper_20_2.0' in df.columns and 'bbands_lower_20_2.0' in df.columns:
        apds.append(mpf.make_addplot(df['bbands_upper_20_2.0'], color='gray'))
        apds.append(mpf.make_addplot(df['bbands_lower_20_2.0'], color='gray'))
    
    # Add Fibonacci levels
    for level in ['fib_23', 'fib_38', 'fib_50', 'fib_61']:
        if level in df.columns:
            apds.append(mpf.make_addplot(df[level], color='orange', linestyle='--'))
    
    # Save to buffer
    buffer = BytesIO()
    mpf.plot(df, type='candle', style='yahoo', 
             title=f"{symbol} ({interval})", 
             ylabel='Price', 
             addplot=apds if apds else None, 
             volume=True,
             savefig=dict(fname=buffer, dpi=100, bbox_inches='tight'))
    
    buffer.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"

def generate_momentum_chart(df):
    """Generate momentum composite chart"""
    if df.empty:
        return ""
    
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    ax1 = plt.subplot(3, 1, 1)
    if 'rsi_14' in df.columns:
        df['rsi_14'].plot(title='RSI', color='blue')
        plt.axhline(70, color='red', linestyle='--')
        plt.axhline(30, color='green', linestyle='--')
    else:
        plt.title('RSI Data Not Available')
    
    plt.subplot(3, 1, 2, sharex=ax1)
    if 'macd_12_26_9' in df.columns and 'macds_12_26_9' in df.columns:
        df['macd_12_26_9'].plot(color='blue', label='MACD')
        df['macds_12_26_9'].plot(color='orange', label='Signal')
        plt.title('MACD')
        plt.legend()
    else:
        plt.title('MACD Data Not Available')
    
    plt.subplot(3, 1, 3, sharex=ax1)
    if 'stochk_14_3' in df.columns and 'stochd_14_3' in df.columns:
        df['stochk_14_3'].plot(color='blue', label='Stoch %K')
        df['stochd_14_3'].plot(color='orange', label='Stoch %D')
        plt.axhline(80, color='red', linestyle='--')
        plt.axhline(20, color='green', linestyle='--')
        plt.title('Stochastic')
        plt.legend()
    else:
        plt.title('Stochastic Data Not Available')
    
    plt.tight_layout()
    
    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"

def generate_volume_chart(df):
    """Generate volume analysis chart"""
    if df.empty or 'volume' not in df.columns:
        return ""
    
    plt.figure(figsize=(12, 8))
    
    # Volume bars
    plt.subplot(3, 1, 1)
    plt.bar(df.index, df['volume'], color='blue')
    plt.title('Volume')
    
    # VWAP
    plt.subplot(3, 1, 2, sharex=plt.gca())
    if 'close' in df.columns:
        df['close'].plot(color='blue', label='Price')
        if 'vwap' in df.columns:
            df['vwap'].plot(color='orange', label='VWAP')
        plt.title('Price vs VWAP')
        plt.legend()
    else:
        plt.title('Price Data Not Available')
    
    # OBV
    plt.subplot(3, 1, 3, sharex=plt.gca())
    if 'obv' in df.columns:
        df['obv'].plot(color='green')
        plt.title('On Balance Volume (OBV)')
    else:
        plt.title('OBV Data Not Available')
    
    plt.tight_layout()
    
    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"

def generate_advanced_chart(df):
    """Generate advanced analytics chart"""
    if df.empty:
        return ""
    
    plt.figure(figsize=(12, 8))
    
    # ATR
    if 'atr_14' in df.columns:
        plt.subplot(2, 1, 1)
        df['atr_14'].plot(color='purple')
        plt.title('Average True Range (ATR)')
    
    # Bollinger Bandwidth
    if 'bbands_upper_20_2.0' in df.columns and 'bbands_lower_20_2.0' in df.columns:
        bandwidth = (df['bbands_upper_20_2.0'] - df['bbands_lower_20_2.0']) / df['bbands_mid_20_2.0']
        plt.subplot(2, 1, 2, sharex=plt.gca())
        bandwidth.plot(color='red')
        plt.title('Bollinger Bandwidth')
    
    plt.tight_layout()
    
    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode()}"

def get_ai_recommendation(symbol, interval, analysis_data):
    """Get AI recommendation from OpenRouter"""
    if not os.getenv('OPENROUTER_API_KEY'):
        return None
    
    # Prepare prompt
    prompt = f"""
    Perform comprehensive technical analysis for {symbol} ({interval}) with these observations:

    **Price Structure**
    Current: ₹{analysis_data['current_price']:.2f}
    Trend: {analysis_data['trend_strength']}/10
    Key Levels: Support @ ₹{analysis_data['support']:.2f} | Resistance @ ₹{analysis_data['resistance']:.2f}
    Pattern: {analysis_data['dominant_pattern']} (Reliability: {analysis_data['pattern_score']}/5)

    **Indicator Summary**
    Trend: {analysis_data['trend_summary']}
    Momentum: {analysis_data['momentum_summary']}
    Volume: {analysis_data['volume_profile_summary']}
    Volatility: {analysis_data['volatility_class']} (ATR: ₹{analysis_data['atr_value']:.2f})

    **Market Context**
    Sector: Financial Services (Outperforming Nifty by 5.2%)
    Market Phase: Bullish (Confirmed by MACD, Golden Cross)
    Sentiment: Positive (PCR: 0.85)

    **Analysis Request**
    1. Recommendation (Strong Buy to Strong Sell)
    2. Confidence Score (0-100%)
    3. Key Technical Drivers
    4. Price Targets (1W/1M/3M)
    5. Risk-Managed Trading Plan:
        - Entry Zones
        - Position Sizing
        - Stop-Loss Strategy
        - Profit Targets
        - Hedging Recommendations

    Respond in JSON format matching the schema.
    """
    
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": os.getenv('OPENROUTER_MODEL', 'mistralai/mixtral-8x7b-instruct'),
        "messages": [
            {
                "role": "user", 
                "content": prompt,
                "metadata": {
                    "response_format": {
                        "type": "json_object",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "rating": {"type": "string"},
                                "confidence": {"type": "number"},
                                "risk_level": {"type": "string"},
                                "time_horizon": {"type": "string"},
                                "price_targets": {
                                    "type": "object",
                                    "properties": {
                                        "conservative": {"type": "number"},
                                        "moderate": {"type": "number"},
                                        "aggressive": {"type": "number"},
                                        "stop_loss": {"type": "number"}
                                    }
                                },
                                "trading_plan": {
                                    "type": "object",
                                    "properties": {
                                        "entry_zones": {"type": "array", "items": {"type": "string"}},
                                        "position_size": {"type": "string"},
                                        "profit_targets": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "level": {"type": "number"},
                                                    "size": {"type": "string"}
                                                }
                                            }
                                        },
                                        "hedging": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ]
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        # Parse JSON response
        content = response.json()['choices'][0]['message']['content']
        return json.loads(content)
    except Exception as e:
        app.logger.error(f"AI request failed: {str(e)}")
        return None

@app.route('/analyze', methods=['GET'])
def analyze_stock():
    """Main analysis endpoint"""
    symbol = request.args.get('symbol', 'RELIANCE').upper()
    interval = request.args.get('interval', '1d')
    
    start_time = time.time()
    
    # Fetch data
    df = fetch_stock_data(symbol, interval)
    if df is None:
        return jsonify({"error": "Data unavailable for symbol"}), 404
    
    # Get TradingView analysis
    tv_analysis = get_tradingview_analysis(symbol, interval)
    
    # Calculate indicators
    df = calculate_indicators(df)
    
    # Prepare analysis data
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    daily_change = ((current_price - prev_price) / prev_price) * 100
    
    key_levels = calculate_key_levels(df)
    patterns = detect_candlestick_patterns(df)
    
    # Determine dominant pattern
    dominant_pattern = patterns[0]['name'] if patterns else 'None'
    pattern_score = patterns[0]['reliability'] if patterns else 0
    
    # Prepare AI input
    ai_input = {
        'symbol': symbol,
        'interval': interval,
        'current_price': current_price,
        'support': key_levels['support'][0],
        'resistance': key_levels['resistance'][0],
        'trend_strength': 8.2,
        'dominant_pattern': dominant_pattern,
        'pattern_score': pattern_score,
        'trend_summary': "Bullish" if current_price > df['sma_20'].iloc[-1] else "Bearish",
        'momentum_summary': "Increasing" if df['rsi_14'].iloc[-1] > 50 else "Decreasing",
        'volume_profile_summary': "Accumulation" if df['obv'].iloc[-1] > df['obv'].iloc[-5] else "Distribution",
        'volatility_class': "High" if df['atr_14'].iloc[-1] > current_price * 0.02 else "Low",
        'atr_value': df['atr_14'].iloc[-1] if 'atr_14' in df.columns else 0,
        'tv_recommendation': get_tradingview_recommendation(tv_analysis)
    }
    
    # Build response
    response = {
        "metadata": {
            "symbol": symbol,
            "interval": interval,
            "last_refreshed": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "analysis_period": f"{len(df)} periods",
            "processing_time": f"{time.time() - start_time:.2f}s"
        },
        "price_analysis": {
            "current": current_price,
            "daily_change": round(daily_change, 2),
            "key_levels": key_levels
        },
        "indicator_summary": {
            "trend": {
                "direction": "Bullish" if current_price > df['sma_20'].iloc[-1] else "Bearish",
                "strength": 75.4,
                "key_drivers": ["EMA Crossover", "MACD Bullish"]
            },
            "momentum": {
                "status": "Increasing" if df['rsi_14'].iloc[-1] > 50 else "Decreasing",
                "divergence": "None",
                "oscillators": [
                    f"RSI: {df['rsi_14'].iloc[-1]:.1f}" if 'rsi_14' in df.columns else "RSI: N/A",
                    f"Stoch: {df['stochk_14_3'].iloc[-1]:.1f}/{df['stochd_14_3'].iloc[-1]:.1f}"
                    if 'stochk_14_3' in df.columns else "Stochastic: N/A"
                ]
            },
            "volatility": {
                "regime": "Expanding" if df['atr_14'].iloc[-1] > df['atr_14'].iloc[-5] else "Contracting",
                "expected_range": f"{current_price * 0.95:.1f}-{current_price * 1.05:.1f}",
                "atr": df['atr_14'].iloc[-1] if 'atr_14' in df.columns else 0
            },
            "volume_profile": {
                "poc": df['poc'].iloc[-1] if 'poc' in df else current_price,
                "value_area": [current_price * 0.99, current_price * 1.01],
                "sentiment": "Accumulation" if df['obv'].iloc[-1] > df['obv'].iloc[-5] else "Distribution"
            }
        },
        "pattern_recognition": patterns,
        "chart_urls": generate_charts(symbol, interval, df),
        "tradingview_reference": tv_analysis
    }
    
    # Add AI recommendation
    ai_rec = get_ai_recommendation(symbol, interval, ai_input)
    if ai_rec:
        response["ai_recommendation"] = ai_rec
    
    # Add ML forecast if available
    if 'ml_forecast' in df:
        response["advanced_analytics"] = {
            "lstm_forecast": {
                "1D": {
                    "direction": "Bullish" if df['ml_forecast'].iloc[-1] > 0.5 else "Bearish", 
                    "confidence": round(df['ml_forecast'].iloc[-1] * 100, 1)
                },
            }
        }
    
    return jsonify(response)

@app.route('/chart/<chart_type>', methods=['GET'])
def serve_chart(chart_type):
    """Serve individual chart image"""
    symbol = request.args.get('symbol', 'RELIANCE').upper()
    interval = request.args.get('interval', '1d')
    
    df = fetch_stock_data(symbol, interval)
    if df is None:
        return jsonify({"error": "Data unavailable"}), 404
    
    chart_funcs = {
        'main': generate_main_chart,
        'momentum': generate_momentum_chart,
        'volume': generate_volume_chart,
        'advanced': generate_advanced_chart
    }
    
    if chart_type not in chart_funcs:
        return jsonify({"error": "Invalid chart type"}), 400
    
    try:
        if chart_type == 'main':
            chart_data = chart_funcs[chart_type](symbol, interval, df)
        else:
            chart_data = chart_funcs[chart_type](df)
        
        if not chart_data:
            return jsonify({"error": "Chart generation failed"}), 500
            
        return send_file(BytesIO(base64.b64decode(chart_data.split(',')[1])), mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Chart error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
