import numpy as np
from datetime import datetime
from .data_loader import DataLoader
from .model import create_model, save_model
from .config import Config


def train():
    # Initialize data loader
    data_loader = DataLoader()

    # Fetch and prepare data
    print("Fetching price and gas data...")
    df = data_loader.fetch_combined_data()

    if df.empty:
        print("Error: No data fetched")
        return

    print(f"Fetched {len(df)} data points")

    # Prepare sequences for LSTM
    print("Preparing sequences...")
    X, y = data_loader.prepare_sequences(df)

    # Split into train/test sets
    train_size = int(len(X) * Config.TRAIN_SPLIT)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # Create and train model
    print("Creating model...")
    model = create_model(
        sequence_length=Config.SEQUENCE_LENGTH, n_features=len(Config.FEATURE_COLUMNS)
    )

    print("Training model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        verbose=1,
    )

    # Evaluate model
    print("\nEvaluating model...")
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss}")

    # Save model
    print("Saving model...")
    model_filename = f"eth_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    save_model(model, model_filename)
    print(f"Model saved as {model_filename}")

    return model, history


if __name__ == "__main__":
    train()
