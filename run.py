from src.train import train
from src.predict import Predictor
from src.config import Config
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="ETH Price Predictor")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--predict", action="store_true", help="Make predictions")
    parser.add_argument(
        "--hours", type=int, default=24, help="Number of hours to predict"
    )

    args = parser.parse_args()

    if args.train:
        print("Training new model...")
        model, history = train()
        print("Training completed!")

    if args.predict:
        model_files = list(Config.MODELS_DIR.glob("*.h5"))
        if not model_files:
            print("No trained models found! Please train a model first.")
            return

        latest_model = str(max(model_files))
        print(f"Using model: {latest_model}")
        predictor = Predictor(latest_model)

        print(f"\nPredicting next {args.hours} hours...")
        predictions = predictor.predict_next_n(args.hours)

        print("\nPredictions:")
        for i, price in enumerate(predictions, 1):
            print(f"Hour {i}: ${price:.2f}")


if __name__ == "__main__":
    main()
