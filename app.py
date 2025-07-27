from flask import Flask, request, jsonify
from tradingview_ta import TA_Handler, Interval, Exchange

app = Flask(__name__)

INTERVAL_MAP = {
    "1m": Interval.INTERVAL_1_MINUTE,
    "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
    "1d": Interval.INTERVAL_1_DAY,
    "1w": Interval.INTERVAL_1_WEEK,
    "1M": Interval.INTERVAL_1_MONTH
}

@app.route('/')
def home():
    return "✅ TradingView TA Flask API for NSE stocks is running!"

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        symbol = data.get("symbol", "").upper()
        interval = data.get("interval", "1d")

        if interval not in INTERVAL_MAP:
            return jsonify({"error": "Invalid interval. Use 1m, 5m, 1h, 1d, etc."}), 400

        handler = TA_Handler(
            symbol=symbol,
            screener="india",
            exchange="NSE",
            interval=INTERVAL_MAP[interval]
        )

        analysis = handler.get_analysis()
        return jsonify({
            "symbol": symbol,
            "interval": interval,
            "price": analysis.indicators.get("close"),
            "RSI": analysis.indicators.get("RSI"),
            "MACD": analysis.indicators.get("MACD.macd"),
            "Signal": analysis.indicators.get("MACD.signal"),
            "summary": analysis.summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
