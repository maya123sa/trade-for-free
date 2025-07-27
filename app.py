from flask import Flask, request, jsonify
from tradingview_ta import TA_Handler, Interval
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app)

# TradingView interval map
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
    return jsonify({"message": "Indian Stock Technical Analyzer API with Moneycontrol"}), 200


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

    # TradingView technical analysis
    handler = TA_Handler(
        symbol=symbol,
        exchange="NSE",
        screener="india",
        interval=INTERVAL_MAP[interval_str]
    )

    try:
        analysis = handler.get_analysis()
        tech_response = {
            "symbol": symbol,
            "interval": interval_str,
            "summary": analysis.summary,
            "indicators": analysis.indicators
        }

        # Add Moneycontrol info
        mc_info = fetch_moneycontrol_info(symbol)
        tech_response["moneycontrol_info"] = mc_info

        return jsonify(tech_response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def fetch_moneycontrol_info(symbol):
    try:
        # This mapping is needed because Moneycontrol uses full company names or slugs
        company_slug = symbol_to_moneycontrol_slug(symbol)
        if not company_slug:
            return {"error": "Company not found in mapping."}

        url = f"https://www.moneycontrol.com/financials/{company_slug}/ratiosVI/{symbol.lower()}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        info = {}

        # Example: scrape PE ratio, ROE, etc.
        pe_row = soup.find("td", string="P/E")
        if pe_row and pe_row.find_next_sibling("td"):
            info["PE Ratio"] = pe_row.find_next_sibling("td").text.strip()

        roe_row = soup.find("td", string="Return On Equity (%)")
        if roe_row and roe_row.find_next_sibling("td"):
            info["ROE"] = roe_row.find_next_sibling("td").text.strip()

        return info if info else {"note": "Basic ratios not found."}

    except Exception as e:
        return {"error": str(e)}


def symbol_to_moneycontrol_slug(symbol):
    # Hardcoded mapping for demo purposes
    slug_map = {
        "RELIANCE": "relianceindustries",
        "TCS": "tataconsultancyservices",
        "INFY": "infosys",
        "HDFCBANK": "hdfcbank",
        "SBIN": "statebankofindia"
        # Add more mappings here
    }
    return slug_map.get(symbol)


if __name__ == '__main__':
    app.run(debug=True)
