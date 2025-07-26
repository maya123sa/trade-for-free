import json
from flask import Flask, request, jsonify
from tradingview_ta import TA_Handler, Interval

app = Flask(__name__)

# Load symbols once
with open("assets/symbols.json") as f:
    SYMBOLS = json.load(f)

INTERVAL_MAP = {
    "1m": Interval.INTERVAL_1_MINUTE,
    "5m": Interval.INTERVAL_5_MINUTES,
    "15m": Interval.INTERVAL_15_MINUTES,
    "1h": Interval.INTERVAL_1_HOUR,
    "4h": Interval.INTERVAL_4_HOURS,
    "1d": Interval.INTERVAL_1_DAY,
    "1w": Interval.INTERVAL_1_WEEK,
    "1mo": Interval.INTERVAL_1_MONTH,
}


@app.route('/')
def home():
    return "Stock Analyzer API is running."


@app.route('/symbols', methods=['GET'])
def get_symbols():
    query = request.args.get('q', '').lower()
    filtered = [s for s in SYMBOLS if query in s['symbol'].lower()]
    return jsonify(filtered)


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    symbol = data.get('symbol')
    interval = data.get('interval')

    if not symbol or not interval:
        return jsonify({"error": "Missing 'symbol' or 'interval'"}), 400

    interval_obj = INTERVAL_MAP.get(interval)
    if not interval_obj:
        return jsonify({"error": "Invalid interval"}), 400

    try:
        handler = TA_Handler(
            symbol=symbol,
            screener="india",
            exchange="NSE",
            interval=interval_obj
        )
        analysis = handler.get_analysis()
        summary = analysis.summary
        indicators = analysis.indicators

        result = {
            "Recommendation": summary.get("RECOMMENDATION", "N/A"),
            "BUY": summary.get("BUY", 0),
            "NEUTRAL": summary.get("NEUTRAL", 0),
            "SELL": summary.get("SELL", 0),
            "Indicators": indicators,
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
