from keras.models import Sequential, load_model as keras_load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from .config import Config


def create_model(sequence_length: int, n_features: int) -> Sequential:
    """
    Create an LSTM model for time series prediction

    Args:
        sequence_length: Number of time steps in each input sequence
        n_features: Number of features in each time step
    """
    model = Sequential(
        [
            # First LSTM layer with return sequences for stacking
            LSTM(
                units=50,
                activation="relu",
                return_sequences=True,
                input_shape=(sequence_length, n_features),
            ),
            Dropout(0.2),
            # Second LSTM layer
            LSTM(units=50, activation="relu"),
            Dropout(0.2),
            # Output layer
            Dense(units=1),
        ]
    )

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    return model


def save_model(model: Sequential, filename: str):
    """Save the trained model"""
    model_path = Config.MODELS_DIR / filename
    model.save(model_path)


def load_model(filename: str) -> Sequential:
    """Load a trained model"""
    model_path = Config.MODELS_DIR / filename
    return keras_load_model(model_path)
