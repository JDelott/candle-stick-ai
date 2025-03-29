import numpy as np
import pandas as pd
from datetime import datetime
from .data_loader import DataLoader
from .model import load_model
from .config import Config
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


class Predictor:
    def __init__(self):
        self.data_loader = DataLoader()
        self.price_scaler = MinMaxScaler()
        self.gas_scaler = MinMaxScaler()

        # Load both models
        models_dir = Path(Config.MODELS_DIR)
        price_models = list(models_dir.glob("eth_price_*.h5"))
        gas_models = list(models_dir.glob("eth_gas_*.h5"))

        if not price_models or not gas_models:
            raise ValueError("Missing required models. Please train first.")

        self.price_model = load_model(str(max(price_models)))
        self.gas_model = load_model(str(max(gas_models)))

    def predict_next_n(self, n: int = 24, target: str = "both") -> dict:
        """Predict next n points for price and/or gas"""
        df = self.data_loader.fetch_combined_data()

        if df.empty:
            raise ValueError("No data fetched")

        results = {}

        # Always use all features for both scalers
        data = df[Config.FEATURE_COLUMNS].values
        self.price_scaler.fit(data)
        self.gas_scaler.fit(data)  # Fit with all features

        if target in ["price", "both"]:
            price_predictions = self._predict_price(df, n)
            results["price"] = price_predictions

            print("\nPrice Predictions (USD):")
            for i, price in enumerate(price_predictions, 1):
                print(f"Hour {i}: ${price:.2f}")

        if target in ["gas", "both"]:
            gas_predictions = self._predict_gas(df, n)
            results["gas"] = gas_predictions

            print("\nGas Predictions (GWEI):")
            for i, gas in enumerate(gas_predictions, 1):
                print(f"Hour {i}: {gas:.1f}")

        return results

    def _predict_price(self, df: pd.DataFrame, n: int) -> list:
        """Price-specific prediction logic"""
        data = df[Config.FEATURE_COLUMNS].values
        scaled_data = self.price_scaler.transform(data)

        X = []
        for i in range(Config.SEQUENCE_LENGTH, len(scaled_data)):
            X.append(scaled_data[i - Config.SEQUENCE_LENGTH : i])
        X = np.array(X)
        current_sequence = X[-1]

        target_idx = Config.FEATURE_COLUMNS.index("close")
        return self._generate_predictions(
            self.price_model,
            current_sequence,
            self.price_scaler,
            target_idx,
            n,
            len(Config.FEATURE_COLUMNS),
        )

    def _predict_gas(self, df: pd.DataFrame, n: int) -> list:
        """Gas-specific prediction logic"""
        # Use all features, just like during training
        data = df[Config.FEATURE_COLUMNS].values
        scaled_data = self.gas_scaler.transform(data)

        X = []
        for i in range(Config.SEQUENCE_LENGTH, len(scaled_data)):
            X.append(scaled_data[i - Config.SEQUENCE_LENGTH : i])
        X = np.array(X)
        current_sequence = X[-1]

        target_idx = Config.FEATURE_COLUMNS.index("gas_fast")
        return self._generate_predictions(
            self.gas_model,
            current_sequence,
            self.gas_scaler,
            target_idx,
            n,
            len(Config.FEATURE_COLUMNS),
        )

    def _generate_predictions(
        self, model, current_sequence, scaler, target_idx, n, n_features
    ):
        """Generate n predictions using the given model and sequence"""
        predictions = []
        sequence = current_sequence.copy()

        # Get current price/gas for reference
        last_pred = scaler.inverse_transform(sequence[-1:].reshape(1, -1))[0][
            target_idx
        ]

        for _ in range(n):
            # Reshape for prediction
            pred_input = sequence.reshape(1, Config.SEQUENCE_LENGTH, -1)
            pred = model.predict(pred_input, verbose=0)

            # Create dummy array for inverse transform
            dummy_array = np.zeros((1, n_features))
            dummy_array[:, target_idx] = pred

            # Get actual prediction value
            actual_pred = scaler.inverse_transform(dummy_array)[0][target_idx]

            if "gas_fast" in Config.FEATURE_COLUMNS[target_idx]:
                # Gas predictions: Allow more volatility (looks good as is)
                volatility = max(
                    0.1, min(0.3, abs(actual_pred - last_pred) / last_pred)
                )
                random_change = np.random.normal(0, volatility * actual_pred)
                actual_pred = max(5, actual_pred + random_change)  # Minimum 5 GWEI
            else:
                # Price predictions: Much stricter limits
                # Limit to 0.2% change per hour (about 5% max per day)
                max_hourly_change = last_pred * 0.002  # 0.2% max change per hour
                actual_pred = max(
                    last_pred - max_hourly_change,
                    min(last_pred + max_hourly_change, actual_pred),
                )

                # Add small random noise to avoid straight lines
                noise = np.random.normal(0, max_hourly_change * 0.1)
                actual_pred = max(
                    last_pred * 0.99, actual_pred + noise
                )  # Prevent big drops

            predictions.append(float(actual_pred))
            last_pred = actual_pred

            # Update sequence
            sequence = np.roll(sequence, -1, axis=0)
            sequence[-1] = scaler.transform([[actual_pred] + [0] * (n_features - 1)])[0]

        return predictions


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
