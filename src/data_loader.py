import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import yfinance as yf
from datetime import datetime, timedelta
from .config import Config

# Load environment variables
load_dotenv()


class DataLoader:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if not self.etherscan_api_key:
            raise ValueError("ETHERSCAN_API_KEY not found in .env file")

    def fetch_gas_data(self) -> dict:
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

            # Print API response for debugging
            print("Etherscan API Response:", data)

            if data["status"] != "1" or "result" not in data:
                print(f"Etherscan API Error: {data.get('message', 'Unknown error')}")
                return None

            return data["result"]

        except Exception as e:
            print(f"Error fetching gas data: {e}")
            return None

    def fetch_combined_data(self) -> pd.DataFrame:
        """Fetch both price and gas data"""
        try:
            # Get price data
            ticker = yf.Ticker("ETH-USD")
            price_df = ticker.history(period="30d", interval="1h")
            price_df = price_df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # Get gas data
            gas_data = self.fetch_gas_data()
            if gas_data:
                # Convert gas prices from string to float and multiply by 10 for GWEI
                price_df["gas_safe_low"] = float(gas_data["SafeGasPrice"]) * 20
                price_df["gas_standard"] = float(gas_data["ProposeGasPrice"]) * 20
                price_df["gas_fast"] = float(gas_data["FastGasPrice"]) * 20
                price_df["gas_base_fee"] = float(gas_data["suggestBaseFee"]) * 20
            else:
                print("Warning: Using default gas values")
                price_df["gas_safe_low"] = 30
                price_df["gas_standard"] = 50
                price_df["gas_fast"] = 70
                price_df["gas_base_fee"] = 40

            # Add other required columns
            price_df["quote_asset_volume"] = 0
            price_df["number_of_trades"] = 0
            price_df["taker_buy_base_asset_volume"] = 0
            price_df["taker_buy_quote_asset_volume"] = 0

            print(f"Current Gas Prices:")
            print(f"Safe Low: {price_df['gas_safe_low'].iloc[-1]:.1f} GWEI")
            print(f"Standard: {price_df['gas_standard'].iloc[-1]:.1f} GWEI")
            print(f"Fast: {price_df['gas_fast'].iloc[-1]:.1f} GWEI")

            return price_df

        except Exception as e:
            print(f"Error in fetch_combined_data: {e}")
            return pd.DataFrame()

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
