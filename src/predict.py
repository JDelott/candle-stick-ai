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

    def calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate price volatility using log returns"""
        returns = np.log(df["close"] / df["close"].shift(1))
        return returns.std() * np.sqrt(24)  # Annualized for hourly data

    def calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calculate price momentum"""
        return df["close"].pct_change(3).mean() * 100  # 3-hour momentum

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
        self, df: pd.DataFrame, n: int = 6, conditions: Dict = None
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Predict ETH price for next n hours"""
        try:
            current_price = df["close"].iloc[-1]
            volatility = self.calculate_volatility(df)
            momentum = self.calculate_momentum(df)

            # Get market trend
            trend_multiplier = 1.0 if conditions.get("price_trend") == "up" else -1.0

            predictions = []
            confidence_intervals = []

            for i in range(n):
                # Combine momentum and trend for drift
                drift = (
                    (momentum / 100) * trend_multiplier * 0.1
                )  # Scale down the effect

                # Add randomness based on historical volatility
                random_change = np.random.normal(
                    drift, volatility / 24
                )  # Hourly volatility

                if i == 0:
                    pred = current_price * (1 + random_change)
                else:
                    pred = predictions[-1] * (1 + random_change)

                predictions.append(pred)

                # Confidence interval widens with time
                interval = pred * (volatility / 24) * (1 + i * 0.1) * 2
                confidence_intervals.append((pred - interval, pred + interval))

            logger.info(f"Price predictions: {[f'${p:.2f}' for p in predictions]}")
            return predictions, confidence_intervals

        except Exception as e:
            logger.error(f"Error in price prediction: {str(e)}")
            raise

    def _predict_gas(
        self, df: pd.DataFrame, n: int = 6, conditions: Dict = None
    ) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Predict gas prices for next n hours"""
        try:
            current_gas = conditions.get("gas_price", df["gas_fast"].iloc[-1])
            logger.info(f"Starting gas prediction from {current_gas:.1f} GWEI")

            # Calculate gas price momentum
            gas_momentum = df["gas_fast"].pct_change(3).mean() * 100

            predictions = []
            confidence_intervals = []

            for i in range(n):
                # Combine trend and momentum
                trend_factor = 1.0 if conditions.get("gas_trend") == "up" else -1.0
                drift = gas_momentum * trend_factor * 0.01

                # Add randomness
                base_change = np.random.normal(drift, 1)

                if i == 0:
                    pred = max(1, current_gas + base_change)
                else:
                    pred = max(1, predictions[-1] + base_change)

                predictions.append(pred)

                # Wider confidence intervals for later predictions
                interval = 2 * (1 + i * 0.1)
                confidence_intervals.append((max(1, pred - interval), pred + interval))

            logger.info(
                f"Gas predictions (GWEI): {[f'{p:.0f}' for p in predictions]}..."
            )
            return predictions, confidence_intervals

        except Exception as e:
            logger.error(f"Error in gas prediction: {str(e)}")
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
