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
    parser.add_argument(
        "--target",
        type=str,
        default="both",
        choices=["price", "gas", "both"],
        help="What to predict: price, gas, or both",
    )

    args = parser.parse_args()

    if args.train:
        print("Training new model...")
        model, history = train()
        print("Training completed!")

    if args.predict:
        print("\nMaking predictions...")
        predictor = Predictor()  # No need to pass model filename anymore
        predictions = predictor.predict_next_n(args.hours, args.target)


if __name__ == "__main__":
    main()
