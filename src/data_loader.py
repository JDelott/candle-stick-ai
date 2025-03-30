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

    def _fetch_price_data(self) -> pd.DataFrame:
        """Fetch ETH price data from Yahoo Finance"""
        try:
            eth = yf.Ticker("ETH-USD")
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)

            df = eth.history(start=start_time, end=end_time, interval="1h")

            # Ensure all required columns exist and are properly named
            required_columns = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }

            # Rename columns to lowercase
            df = df.rename(columns=required_columns)

            # Add missing trading columns with placeholder data
            df["quote_asset_volume"] = df["volume"] * df["close"]
            df["number_of_trades"] = np.random.randint(1000, 5000, size=len(df))
            df["taker_buy_base_asset_volume"] = df["volume"] * 0.6
            df["taker_buy_quote_asset_volume"] = df["quote_asset_volume"] * 0.6

            logger.info(f"Price data shape: {df.shape}")
            logger.info(f"Price columns: {df.columns.tolist()}")

            return df

        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            logger.error(
                f"DataFrame info: {df.info()}"
                if "df" in locals()
                else "No DataFrame created"
            )
            raise

    def _fetch_gas_data(self) -> dict:
        """Fetch current gas price from Etherscan"""
        try:
            url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={self.etherscan_api_key}"
            response = requests.get(url)
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching gas data: {str(e)}")
            raise

    def fetch_combined_data(self) -> pd.DataFrame:
        """Fetch both historical price and gas data"""
        try:
            # Get ETH price history
            df = self._fetch_price_data()

            # Get current gas price
            gas_data = self._fetch_gas_data()

            if gas_data["status"] != "1" or "result" not in gas_data:
                raise ValueError(
                    f"Etherscan API Error: {gas_data.get('message', 'Unknown error')}"
                )

            # Create gas price data points
            current_gas = float(gas_data["result"]["FastGasPrice"])
            logger.info(f"Current gas price: {current_gas} GWEI")

            # Create gas variations
            base_gas = max(20, current_gas)  # Ensure minimum gas price
            timestamps = df.index
            gas_variations = np.random.normal(0, 2, len(timestamps))

            # Add gas columns
            df["gas_fast"] = base_gas + gas_variations
            df["gas_safe_low"] = (
                float(gas_data["result"]["SafeGasPrice"]) + gas_variations * 0.8
            )
            df["gas_standard"] = (
                float(gas_data["result"]["ProposeGasPrice"]) + gas_variations * 0.9
            )
            df["gas_base_fee"] = (
                float(gas_data["result"]["suggestBaseFee"]) + gas_variations * 0.7
            )

            # Ensure no negative gas prices
            gas_columns = ["gas_fast", "gas_safe_low", "gas_standard", "gas_base_fee"]
            for col in gas_columns:
                df[col] = df[col].clip(lower=1)

            logger.info(f"Final DataFrame shape: {df.shape}")
            logger.info(f"Final columns: {df.columns.tolist()}")

            return df

        except Exception as e:
            logger.error(f"Error in fetch_combined_data: {str(e)}")
            raise

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
