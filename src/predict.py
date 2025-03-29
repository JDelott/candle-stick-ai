import numpy as np
from datetime import datetime
from .data_loader import DataLoader
from .model import load_model
from .config import Config


class Predictor:
    def __init__(self, model_filename: str):
        self.data_loader = DataLoader()
        self.model = load_model(model_filename)

    def predict_next(self, target="price"):
        """Predict the next point for either price or gas"""
        df = self.data_loader.fetch_combined_data()

        if df.empty:
            raise ValueError("No data fetched")

        # Set target column based on what we're predicting
        Config.TARGET_COLUMN = "close" if target == "price" else "gas_fast"

        # Get original scale for inverse transform
        original_data = df[Config.FEATURE_COLUMNS].values
        self.data_loader.scaler.fit(original_data)

        # Prepare sequence
        X, _ = self.data_loader.prepare_sequences(df)

        # Get the most recent sequence
        last_sequence = X[-1:]

        # Make prediction
        prediction = self.model.predict(last_sequence, verbose=0)

        # Inverse transform to get actual price
        dummy_array = np.zeros((1, len(Config.FEATURE_COLUMNS)))
        dummy_array[:, Config.FEATURE_COLUMNS.index(Config.TARGET_COLUMN)] = prediction
        actual_prediction = self.data_loader.scaler.inverse_transform(dummy_array)[0][
            Config.FEATURE_COLUMNS.index(Config.TARGET_COLUMN)
        ]

        return float(actual_prediction)

    def predict_next_n(self, n: int = 24) -> list:
        """Predict next n price points"""
        predictions = []
        df = self.data_loader.fetch_candlestick_data()

        if df.empty:
            raise ValueError("No data fetched")

        # Get original scale for inverse transform
        original_data = df[Config.FEATURE_COLUMNS].values
        self.data_loader.scaler.fit(original_data)

        # Prepare initial sequence
        X, _ = self.data_loader.prepare_sequences(df)
        current_sequence = X[-1]

        # Make n predictions
        for _ in range(n):
            sequence = current_sequence.reshape(1, Config.SEQUENCE_LENGTH, -1)
            pred = self.model.predict(sequence, verbose=0)

            # Inverse transform to get actual price
            dummy_array = np.zeros((1, len(Config.FEATURE_COLUMNS)))
            dummy_array[:, Config.FEATURE_COLUMNS.index(Config.TARGET_COLUMN)] = pred
            actual_pred = self.data_loader.scaler.inverse_transform(dummy_array)[0][
                Config.FEATURE_COLUMNS.index(Config.TARGET_COLUMN)
            ]

            predictions.append(float(actual_pred))

            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred[0]

        return predictions

    def predict_both(self):
        """Predict both price and gas"""
        price_pred = self.predict_next(target="price")
        gas_pred = self.predict_next(target="gas")

        return {"price": price_pred, "gas": gas_pred}


def main():
    # Use the most recent model in the models directory
    import glob

    model_files = glob.glob(str(Config.MODELS_DIR / "*.h5"))
    if not model_files:
        print("No trained models found!")
        return

    latest_model = max(model_files)
    predictor = Predictor(latest_model)

    # Make predictions
    try:
        next_price = predictor.predict_next()
        print(f"\nNext price prediction: {next_price:.2f}")

        future_prices = predictor.predict_next_n(24)
        print("\nNext 24 hours predictions:")
        for i, price in enumerate(future_prices, 1):
            print(f"Hour {i}: {price:.2f}")

    except Exception as e:
        print(f"Error making predictions: {e}")


if __name__ == "__main__":
    main()
