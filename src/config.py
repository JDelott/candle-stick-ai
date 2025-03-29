from pathlib import Path
from datetime import datetime


class Config:
    # Project structure
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    MODELS_DIR = ROOT_DIR / "models"

    # Ensure directories exist
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # Minimal training parameters for quick results
    SEQUENCE_LENGTH = 10  # Reduced from 15
    BATCH_SIZE = 32
    EPOCHS = 3  # Reduced from 5
    TRAIN_SPLIT = 0.8

    # Feature columns including both price and gas
    FEATURE_COLUMNS = [
        # Price data
        "open",
        "high",
        "low",
        "close",
        "volume",
        # Gas data
        "gas_safe_low",
        "gas_standard",
        "gas_fast",
        "gas_base_fee",
        # Other features
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]

    # Default target (can be changed during prediction)
    TARGET_COLUMN = "close"  # or "gas_fast" for gas prediction

    # Trading parameters
    SYMBOL = "ETHUSDT"
    TIMEFRAME = "1h"

    # Model filename
    MODEL_FILENAME = f"eth_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
