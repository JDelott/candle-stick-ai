import numpy as np
from datetime import datetime
from .data_loader import DataLoader
from .model import create_model, save_model
from .config import Config


def train():
    # Initialize data loader
    data_loader = DataLoader()

    # Fetch data
    print("Fetching price and gas data...")
    df = data_loader.fetch_combined_data()

    if df.empty:
        print("Error: No data fetched")
        return

    print(f"Fetched {len(df)} data points")

    # Train price model
    print("\nTraining price model...")
    Config.TARGET_COLUMN = "close"
    price_model = create_model(
        sequence_length=Config.SEQUENCE_LENGTH, n_features=len(Config.FEATURE_COLUMNS)
    )

    X, y = data_loader.prepare_sequences(df)
    price_history = price_model.fit(
        X,
        y,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    # Train gas model
    print("\nTraining gas model...")
    Config.TARGET_COLUMN = "gas_fast"
    gas_model = create_model(
        sequence_length=Config.SEQUENCE_LENGTH, n_features=len(Config.FEATURE_COLUMNS)
    )

    X, y = data_loader.prepare_sequences(df)
    gas_history = gas_model.fit(
        X,
        y,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_split=0.2,
        verbose=1,
    )

    # Save models
    print("\nSaving models...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(price_model, f"eth_price_{timestamp}.h5")
    save_model(gas_model, f"eth_gas_{timestamp}.h5")

    return (price_model, gas_model), (price_history, gas_history)


if __name__ == "__main__":
    train()
