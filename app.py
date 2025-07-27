from flask import Flask, request, jsonify
from tradingview_ta import TA_Handler, Interval
from flask_cors import CORS  # <-- Add this import

app = Flask(__name__)
CORS(app)  # <-- Enable CORS for all routes

# Map user input to tradingview-ta intervals
INTERVAL_MAP = {
    "1m": Interval.INTERVAL_1_MINUTE,
    "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "30m": Interval.INTERVAL_30_MINUTES,
    "1h": Interval.INTERVAL_1_HOUR,
    "2h": Interval.INTERVAL_2_HOURS,
    "4h": Interval.INTERVAL_4_HOURS,
    "1d": Interval.INTERVAL_1_DAY,
    "1w": Interval.INTERVAL_1_WEEK,
    "1M": Interval.INTERVAL_1_MONTH,
}

@app.route('/')
def home():
    return jsonify({"message": "Indian Stock Technical Analyzer API"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()

    symbol = data.get("symbol", "").strip().upper()
    interval_str = data.get("interval", "").strip()

    if not symbol or interval_str not in INTERVAL_MAP:
        return jsonify({
            "error": "Please provide a valid 'symbol' and supported 'interval'.",
            "supported_intervals": list(INTERVAL_MAP.keys())
        }), 400

    handler = TA_Handler(
        symbol=symbol,
        exchange="NSE",
        screener="india",
        interval=INTERVAL_MAP[interval_str]
    )

    try:
        analysis = handler.get_analysis()
        response = {
            "symbol": symbol,
            "interval": interval_str,
            "summary": analysis.summary,
            "indicators": analysis.indicators
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
