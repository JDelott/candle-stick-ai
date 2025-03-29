from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from .predict import Predictor
from .data_loader import DataLoader
import traceback
import logging
import numpy as np

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize predictor once
predictor = Predictor()

# Enable CORS for specific routes
CORS(app, resources={r"/predict": {"origins": "*"}, r"/historical": {"origins": "*"}})


@app.route("/predict", methods=["GET"])
def get_predictions():
    try:
        logger.info("Starting prediction request...")

        # Get predictions with error handling
        try:
            results = predictor.predict_next_n(n=24, target="both")
            logger.info("Raw predictions generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate predictions: {str(e)}")
            logger.error(traceback.format_exc())
            return (
                jsonify({"success": False, "error": "Failed to generate predictions"}),
                500,
            )

        # Format predictions with error handling
        try:
            predictions = []
            for i in range(24):
                prediction = {"hour": f"Hour {i+1}", "price": None, "gas": None}

                if "price" in results and len(results["price"]) > i:
                    price = results["price"][i]
                    # Format price to 2 decimal places
                    prediction["price"] = (
                        round(float(price), 2) if price is not None else None
                    )

                if "gas" in results and len(results["gas"]) > i:
                    gas = results["gas"][i]
                    # Convert gas to integer GWEI
                    prediction["gas"] = round(float(gas)) if gas is not None else None

                predictions.append(prediction)

            response_data = {
                "success": True,
                "predictions": predictions,
                "market_conditions": {
                    "sentiment": round(results["market_conditions"]["sentiment"], 2),
                    "volatility": round(results["market_conditions"]["volatility"], 2),
                    "volume_trend": round(
                        results["market_conditions"]["volume_trend"], 2
                    ),
                },
            }

            logger.info(f"Sending {len(predictions)} predictions")
            logger.debug(f"First prediction: {predictions[0]}")

            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Failed to format predictions: {str(e)}")
            logger.error(traceback.format_exc())
            return (
                jsonify({"success": False, "error": "Failed to format predictions"}),
                500,
            )

    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": "Unexpected error occurred"}), 500


@app.route("/historical", methods=["GET"])
def get_historical():
    try:
        data_loader = DataLoader()
        df = data_loader.fetch_combined_data()

        if df.empty:
            raise ValueError("No historical data available")

        # Convert DataFrame to list of dictionaries
        historical = []
        for index, row in df.iterrows():
            historical.append(
                {
                    "timestamp": index.isoformat(),
                    "price": float(row["close"]),
                    "gas": float(row["gas_fast"]),
                }
            )

        return jsonify({"success": True, "historical": historical})
    except Exception as e:
        print(f"Error fetching historical data: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
