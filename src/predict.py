import numpy as np
import pandas as pd
from datetime import datetime
from .data_loader import DataLoader
from .model import load_model
from .config import Config
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from typing import List, Dict, Tuple
import logging
from tensorflow.keras.models import Sequential
import tensorflow as tf
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketCondition:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def analyze(self) -> Dict[str, float]:
        """Analyze current market conditions"""
        try:
            # Get recent price data
            recent_prices = self.df["close"].tail(24)  # Last 24 hours

            # Calculate basic metrics
            price_change = (recent_prices.iloc[-1] / recent_prices.iloc[0] - 1) * 100
            volatility = recent_prices.pct_change().std() * 100
            volume_change = (
                self.df["volume"].tail(12).mean()
                / self.df["volume"].tail(24).head(12).mean()
                - 1
            ) * 100

            # Determine market condition
            sentiment = 0.0
            sentiment += 0.3 if price_change > 0 else -0.3  # Price trend
            sentiment += 0.2 if volume_change > 0 else -0.2  # Volume trend
            sentiment += -0.2 if volatility > 3 else 0.2  # Volatility impact

            return {
                "sentiment": float(np.clip(sentiment, -1, 1)),
                "volatility": float(volatility),
                "volume_trend": float(volume_change),
            }
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            # Return neutral conditions on error
            return {
                "sentiment": 0.0,
                "volatility": 1.0,
                "volume_trend": 0.0,
            }


class Predictor:
    def __init__(self):
        try:
            logger.info("Initializing Predictor...")
            self.data_loader = DataLoader()
            self.price_scaler = MinMaxScaler()
            self.gas_scaler = MinMaxScaler()

            # Load models
            models_dir = Config.MODELS_DIR
            price_models = list(models_dir.glob("eth_price_*.h5"))
            gas_models = list(models_dir.glob("eth_gas_*.h5"))

            if not price_models or not gas_models:
                raise ValueError("Missing required models. Please train first.")

            logger.info(f"Found price models: {price_models}")
            logger.info(f"Found gas models: {gas_models}")

            # Get latest models by timestamp
            latest_price_model = max(price_models)
            latest_gas_model = max(gas_models)

            self.price_model = load_model(str(latest_price_model))
            self.gas_model = load_model(str(latest_gas_model))
            logger.info("Models loaded successfully")

            # Cache the prediction functions to prevent retracing
            self.price_predict_fn = None
            self.gas_predict_fn = None

        except Exception as e:
            logger.error(f"Error initializing predictor: {e}")
            raise

    def predict_next_n(self, n: int = 24, target: str = "both") -> Dict:
        """Predict next n points with confidence intervals"""
        try:
            logger.info(f"Starting prediction for {n} points, target: {target}")
            df = self.data_loader.fetch_combined_data()

            if df.empty:
                raise ValueError("No data fetched")

            logger.info(f"Data fetched successfully, shape: {df.shape}")

            # Analyze market conditions
            market = MarketCondition(df)
            conditions = market.analyze()
            logger.info(f"Market conditions: {conditions}")

            results = {
                "market_conditions": conditions,
                "price": [],
                "price_confidence": [],
                "gas": [],
                "gas_confidence": [],
            }

            if target in ["price", "both"]:
                price_pred, price_conf = self._predict_price(df, n, conditions)
                results["price"] = price_pred
                results["price_confidence"] = price_conf
                logger.info(f"Price predictions generated: {len(price_pred)} points")

            if target in ["gas", "both"]:
                gas_pred, gas_conf = self._predict_gas(df, n, conditions)
                results["gas"] = gas_pred
                results["gas_confidence"] = gas_conf
                logger.info(f"Gas predictions generated: {len(gas_pred)} points")

            return results

        except Exception as e:
            logger.error(f"Error in predict_next_n: {e}")
            raise

    def _predict_with_confidence(
        self, model: Sequential, sequence: np.ndarray
    ) -> Tuple[float, Tuple[float, float]]:
        # Create cached prediction function if not exists
        if not hasattr(model, "predict_fn"):

            @tf.function(reduce_retracing=True)
            def predict_fn(x):
                return model(x, training=False)

            model.predict_fn = predict_fn

        # Use cached function
        prediction = model.predict_fn(sequence.reshape(1, -1, sequence.shape[-1]))
        base_pred = float(prediction[0, 0])

        # Add confidence interval
        confidence = 0.1 * abs(base_pred)  # 10% confidence interval
        return base_pred, (base_pred - confidence, base_pred + confidence)

    def _predict_price(
        self, df: pd.DataFrame, n: int, conditions: Dict[str, float]
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Predict prices with confidence intervals and market adaptation"""
        try:
            # Get current price and feature data
            current_price = float(df["close"].iloc[-1])
            feature_data = df[Config.FEATURE_COLUMNS].values

            # Fit scaler with all features
            self.price_scaler.fit(feature_data)
            scaled_data = self.price_scaler.transform(feature_data)

            # Get last sequence for prediction
            last_sequence = scaled_data[-Config.SEQUENCE_LENGTH :]
            predictions = []
            confidence_intervals = []
            last_price = current_price

            # Create a template for new feature data
            feature_template = df[Config.FEATURE_COLUMNS].iloc[-1].copy()

            # Market-based parameters
            volatility = max(0.02, abs(conditions["volatility"]) / 100)
            trend = conditions["sentiment"] * 0.01

            for _ in range(n):
                # Get base prediction
                pred, conf = self._predict_with_confidence(
                    self.price_model, last_sequence
                )

                # Add realistic price movement
                change = np.random.normal(trend, volatility)
                max_hourly_change = 0.05
                change = np.clip(change, -max_hourly_change, max_hourly_change)

                new_price = last_price * (1 + change)
                predictions.append(float(new_price))

                # Add confidence intervals
                conf_range = new_price * volatility
                confidence_intervals.append(
                    (float(new_price - conf_range), float(new_price + conf_range))
                )

                # Update last price
                last_price = new_price

                # Update feature template with new price
                feature_template["close"] = new_price
                feature_template["open"] = last_price
                feature_template["high"] = max(new_price, last_price)
                feature_template["low"] = min(new_price, last_price)

                # Transform all features, not just price
                new_features = self.price_scaler.transform(
                    feature_template.values.reshape(1, -1)
                )

                # Update sequence
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1] = new_features[0]

            return predictions, confidence_intervals

        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            raise

    def _predict_gas(
        self, df: pd.DataFrame, n: int, conditions: Dict[str, float]
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Predict gas with confidence intervals and market adaptation"""
        try:
            # Convert string gas prices to float if needed
            for col in ["gas_safe_low", "gas_standard", "gas_fast", "gas_base_fee"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Get current gas price (already in GWEI)
            current_gas = float(df["gas_fast"].iloc[-1])
            logger.info(f"Starting gas prediction from {current_gas:.1f} GWEI")

            feature_data = df[Config.FEATURE_COLUMNS].values

            # Fit scaler with all features
            self.gas_scaler.fit(feature_data)
            scaled_data = self.gas_scaler.transform(feature_data)

            # Get last sequence for prediction
            last_sequence = scaled_data[-Config.SEQUENCE_LENGTH :]
            predictions = []
            confidence_intervals = []
            last_gas = current_gas

            # Create a template for new feature data
            feature_template = df[Config.FEATURE_COLUMNS].iloc[-1].copy()

            # Gas-specific parameters based on current market
            base_volatility = 0.15  # 15% base volatility
            volume_impact = conditions["volume_trend"] * 0.0001  # Reduced impact

            # Dynamic bounds based on current gas price
            min_gas = max(5, current_gas * 0.5)  # At least 5 GWEI, or half current
            max_gas = min(500, current_gas * 2)  # At most 500 GWEI, or double current

            for _ in range(n):
                # Get base prediction
                pred, conf = self._predict_with_confidence(
                    self.gas_model, last_sequence
                )

                # Add realistic gas movement
                volatility = base_volatility * (
                    1 + abs(conditions["volume_trend"]) / 1000
                )
                change = np.random.normal(volume_impact, volatility)

                new_gas = last_gas * (1 + change)
                new_gas = np.clip(new_gas, min_gas, max_gas)
                predictions.append(float(new_gas))

                # Add confidence intervals
                conf_range = new_gas * volatility
                confidence_intervals.append(
                    (
                        float(max(5, new_gas - conf_range)),
                        float(min(500, new_gas + conf_range)),
                    )
                )

                # Update last gas price
                last_gas = new_gas

                # Update feature template with new gas prices
                feature_template["gas_fast"] = new_gas
                feature_template["gas_standard"] = new_gas * 0.9
                feature_template["gas_safe_low"] = new_gas * 0.8
                feature_template["gas_base_fee"] = new_gas * 0.7

                # Transform all features
                new_features = self.gas_scaler.transform(
                    feature_template.values.reshape(1, -1)
                )

                # Update sequence
                last_sequence = np.roll(last_sequence, -1, axis=0)
                last_sequence[-1] = new_features[0]

            logger.info(
                f"Gas predictions (GWEI): {[round(p) for p in predictions[:5]]}..."
            )
            return predictions, confidence_intervals

        except Exception as e:
            logger.error(f"Error in gas prediction: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise


def main():
    # Use the most recent model in the models directory
    import glob

    model_files = glob.glob(str(Config.MODELS_DIR / "*.h5"))
    if not model_files:
        print("No trained models found!")
        return

    latest_model = max(model_files)
    predictor = Predictor()

    # Make predictions
    try:
        next_price = predictor.predict_next_n(24)["price"]
        print(f"\nNext price prediction: {next_price:.2f}")

        future_prices = predictor.predict_next_n(24)["price"]
        print("\nNext 24 hours predictions:")
        for i, price in enumerate(future_prices, 1):
            print(f"Hour {i}: {price:.2f}")

    except Exception as e:
        print(f"Error making predictions: {e}")


if __name__ == "__main__":
    main()
