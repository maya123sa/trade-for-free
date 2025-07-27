from flask import Flask, request, jsonify
from tradingview_ta import TA_Handler, Interval

app = Flask(__name__)

# Valid intervals
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


# Optional manual testing via console
def run_manual_test():
    symbol = input("Enter stock symbol (e.g., TATAMOTORS.NS): ").strip()
    interval = input("Enter interval (e.g., 1d, 5m, 1h): ").strip()

    interval_obj = INTERVAL_MAP.get(interval)
    if not interval_obj:
        print("❌ Invalid interval.")
        return

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

        print("\n📊 Analysis Result:")
        print("Recommendation:", summary.get("RECOMMENDATION", "N/A"))
        print("BUY:", summary.get("BUY", 0))
        print("NEUTRAL:", summary.get("NEUTRAL", 0))
        print("SELL:", summary.get("SELL", 0))
        print("\n🔍 Indicators:")
        for key, value in indicators.items():
            print(f"{key}: {value}")

    except Exception as e:
        print("❌ Error:", str(e))


if __name__ == '__main__':
    # Uncomment below if you want to run the API server
    app.run(debug=True)

    # Or uncomment below if you want to test it from terminal directly
    # run_manual_test()
