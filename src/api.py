from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from .predict import Predictor
from .data_loader import DataLoader
import traceback
import logging
import numpy as np
from datetime import datetime, timedelta
import random
from .services.chat_service import ChatService

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize predictor once
predictor = Predictor()

# Enable CORS for specific routes
CORS(
    app,
    resources={
        r"/predict": {"origins": "*"},
        r"/historical": {"origins": "*"},
        r"/market-conditions": {"origins": "*"},
    },
)

chat_service = ChatService()


@app.route("/predict", methods=["GET"])
def get_predictions():
    try:
        # Get current timestamp
        current_time = datetime.now()

        # Generate 24 hourly timestamps
        predictions = []
        for i in range(24):
            future_time = current_time + timedelta(hours=i)
            hour = future_time.strftime("%Y-%m-%d %H:00")

            # For now, generate some placeholder predictions
            # You'll replace these with actual ML predictions later
            predictions.append(
                {
                    "hour": hour,
                    "price": round(
                        random.uniform(2000, 3000), 2
                    ),  # Random ETH price between $2000-$3000
                    "gas": round(
                        random.uniform(30, 100), 1
                    ),  # Random gas price between 30-100 GWEI
                }
            )

        return jsonify({"success": True, "predictions": predictions})
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/historical/<type>", methods=["GET"])
def get_historical_by_type(type):
    print(f"Received request for historical {type} data")  # Add logging

    if type not in ["price", "gas"]:
        return (
            jsonify(
                {"success": False, "error": "Invalid type. Must be 'price' or 'gas'"}
            ),
            400,
        )

    try:
        data_loader = DataLoader()
        df = data_loader.fetch_combined_data()

        if df.empty:
            print("No data returned from data_loader")  # Add logging
            raise ValueError("No historical data available")

        # Convert DataFrame to list of dictionaries with only requested type
        historical = []
        for index, row in df.iterrows():
            try:
                value = (
                    float(row["close"]) if type == "price" else float(row["gas_fast"])
                )
                historical.append({"timestamp": index.isoformat(), "value": value})
            except Exception as e:
                print(f"Error processing row: {e}")  # Add logging
                continue

        print(f"Returning {len(historical)} data points")  # Add logging
        return jsonify({"success": True, "data": historical})
    except Exception as e:
        print(f"Error fetching historical {type} data: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/market-conditions", methods=["GET"])
def get_market_conditions():
    try:
        # For now, generate some placeholder market conditions
        # You'll replace these with actual calculations later
        conditions = {
            "sentiment": round(random.uniform(-1, 1), 2),  # -1 to 1
            "volatility": round(random.uniform(0, 1), 2),  # 0 to 1
            "volume_trend": round(random.uniform(-1, 1), 2),  # -1 to 1
        }

        return jsonify({"success": True, "conditions": conditions})
    except Exception as e:
        print(f"Error generating market conditions: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        question = data.get("question")

        if not question:
            return jsonify({"success": False, "error": "No question provided"}), 400

        # Get current market data
        data_loader = DataLoader()
        df = data_loader.fetch_combined_data()

        market_data = {
            "eth_price": float(df["close"].iloc[-1]),
            "gas_price": float(df["gas_fast"].iloc[-1]),
            "price_trend": (
                "up" if df["close"].iloc[-1] > df["close"].iloc[-2] else "down"
            ),
            "gas_trend": (
                "up" if df["gas_fast"].iloc[-1] > df["gas_fast"].iloc[-2] else "down"
            ),
        }

        response = chat_service.get_response(question, market_data)
        return jsonify(response)

    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
