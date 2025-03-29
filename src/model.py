from keras.models import Sequential, load_model as keras_load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from .config import Config
from pathlib import Path


def create_model(
    sequence_length: int, n_features: int, target: str = "price"
) -> Sequential:
    """
    Create an LSTM model for time series prediction

    Args:
        sequence_length: Number of time steps in each input sequence
        n_features: Number of features in each time step
    """
    if target == "gas":
        # Enhanced model for gas predictions
        model = Sequential(
            [
                LSTM(
                    units=64,
                    activation="relu",
                    return_sequences=True,
                    input_shape=(sequence_length, n_features),
                ),
                Dropout(0.1),
                LSTM(units=32, activation="relu", return_sequences=True),
                Dropout(0.1),
                LSTM(units=16, activation="relu"),
                Dense(units=8, activation="relu"),
                Dense(units=1),
            ]
        )
        # Use mean absolute error for gas (better for spiky data)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mae")
    else:
        # Original model for price predictions
        model = Sequential(
            [
                LSTM(
                    units=50,
                    activation="relu",
                    return_sequences=True,
                    input_shape=(sequence_length, n_features),
                ),
                Dropout(0.2),
                LSTM(units=50, activation="relu"),
                Dropout(0.2),
                Dense(units=1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    return model


def save_model(model: Sequential, filename: str):
    """Save the trained model"""
    model_path = Config.MODELS_DIR / filename
    model.save(model_path)


def load_model(filename: str, target: str = "price") -> Sequential:
    """Load a trained model"""
    # Get just the filename without the path
    model_filename = Path(filename).name
    model_path = Config.MODELS_DIR / model_filename
    return keras_load_model(str(model_path))
