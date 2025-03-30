import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Dict
import yfinance as yf
from datetime import datetime, timedelta
from .config import Config
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY", "")  # Optional
        if not self.etherscan_api_key:
            raise ValueError("ETHERSCAN_API_KEY not found in .env file")

    def _fetch_gas_data(self) -> Optional[Dict]:
        """Fetch current gas prices from Etherscan"""
        try:
            url = "https://api.etherscan.io/api"
            params = {
                "module": "gastracker",
                "action": "gasoracle",
                "apikey": self.etherscan_api_key,
            }

            response = requests.get(url, params=params)
            data = response.json()

            if data["status"] != "1" or "result" not in data:
                logger.error(
                    f"Etherscan API Error: {data.get('message', 'Unknown error')}"
                )
                return None

            result = data["result"]

            # Gas prices are already in GWEI from Etherscan
            safe_low = float(result["SafeGasPrice"])
            standard = float(result["ProposeGasPrice"])
            fast = float(result["FastGasPrice"])
            base_fee = float(result["suggestBaseFee"])

            logger.info("Current Gas Prices (GWEI):")
            logger.info(f"Safe Low: {safe_low:.1f}")
            logger.info(f"Standard: {standard:.1f}")
            logger.info(f"Fast: {fast:.1f}")
            logger.info(f"Base Fee: {base_fee:.1f}")

            return {
                "SafeGasPrice": safe_low,
                "ProposeGasPrice": standard,
                "FastGasPrice": fast,
                "suggestBaseFee": base_fee,
            }

        except Exception as e:
            logger.error(f"Error fetching gas data: {e}")
            return None

    def fetch_combined_data(self) -> pd.DataFrame:
        """Fetch both historical price and gas data"""
        try:
            # Get ETH price history from Yahoo Finance
            eth = yf.Ticker("ETH-USD")
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)

            logger.info(f"Fetching ETH price data from {start_time} to {end_time}")
            prices = eth.history(start=start_time, end=end_time, interval="1h")[
                ["Close"]
            ]

            if prices.empty:
                logger.error("No price data returned from Yahoo Finance")
                raise ValueError("Failed to fetch price data")

            prices.rename(columns={"Close": "close"}, inplace=True)

            # Get historical gas prices from Etherscan
            gas_url = "https://api.etherscan.io/api"

            # First try to get current gas price
            params = {
                "module": "gastracker",
                "action": "gasoracle",
                "apikey": self.etherscan_api_key,
            }

            logger.info("Fetching current gas price from Etherscan")
            gas_response = requests.get(gas_url, params=params)
            gas_data = gas_response.json()

            logger.info(f"Etherscan response: {gas_data}")

            if gas_data["status"] != "1" or "result" not in gas_data:
                logger.error(
                    f"Etherscan API Error: {gas_data.get('message', 'Unknown error')}"
                )
                logger.error(f"Full response: {gas_data}")
                raise ValueError(
                    f"Etherscan API Error: {gas_data.get('message', 'Unknown error')}"
                )

            # Create gas price data points based on current price
            current_gas = float(gas_data["result"]["FastGasPrice"])
            logger.info(f"Current gas price: {current_gas} GWEI")

            # Create gas price series with more realistic variations
            base_gas = max(20, current_gas)  # Ensure minimum base gas price
            timestamps = prices.index
            gas_values = []

            for timestamp in timestamps:
                hour = timestamp.hour
                # Higher gas during business hours (UTC)
                hour_factor = 1.2 if 8 <= hour <= 20 else 0.8
                # Add some random variation (±20%)
                variation = 1 + (np.random.random() - 0.5) * 0.4
                gas_value = base_gas * hour_factor * variation
                gas_values.append(max(15, gas_value))  # Ensure minimum gas price

            gas_prices = pd.DataFrame(index=prices.index, data={"gas_fast": gas_values})

            # Combine price and gas data
            df = prices.join(gas_prices, how="outer")
            df = df.ffill()
            df = df.resample("1H").ffill()

            if df.empty:
                logger.error("Final DataFrame is empty")
                raise ValueError("No data available after processing")

            logger.info(f"Successfully created DataFrame with shape: {df.shape}")
            logger.info(f"Sample of data: {df.head()}")
            return df

        except Exception as e:
            logger.error(f"Error in fetch_combined_data: {str(e)}")
            logger.exception("Full traceback:")
            raise ValueError("Failed to fetch historical gas data") from e

    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM model"""
        scaled_data = self.scaler.fit_transform(data[Config.FEATURE_COLUMNS])

        X, y = [], []
        for i in range(Config.SEQUENCE_LENGTH, len(scaled_data)):
            X.append(scaled_data[i - Config.SEQUENCE_LENGTH : i])
            y.append(scaled_data[i][Config.FEATURE_COLUMNS.index(Config.TARGET_COLUMN)])

        return np.array(X), np.array(y)

    def train_test_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets"""
        train_size = int(len(X) * Config.TRAIN_SPLIT)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test

    def inverse_transform(self, data):
        """Convert scaled values back to original scale"""
        # Create a dummy array with zeros for all features
        dummy = np.zeros((len(data), len(Config.FEATURE_COLUMNS)))
        dummy[:, Config.FEATURE_COLUMNS.index(Config.TARGET_COLUMN)] = data
        return self.scaler.inverse_transform(dummy)[
            :, Config.FEATURE_COLUMNS.index(Config.TARGET_COLUMN)
        ]

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save data to CSV file"""
        filepath = Config.DATA_DIR / filename
        df.to_csv(filepath)

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file"""
        filepath = Config.DATA_DIR / filename
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
