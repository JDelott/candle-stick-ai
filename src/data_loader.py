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

    def fetch_combined_data(self) -> pd.DataFrame:
        """Fetch both price and gas data"""
        try:
            # 1. Get price data from Yahoo
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

            # 2. Get gas data if API key is available
            if self.etherscan_api_key:
                try:
                    gas_url = "https://api.etherscan.io/api"
                    gas_params = {
                        "module": "gastracker",
                        "action": "gasOracle",
                        "apikey": self.etherscan_api_key,
                    }

                    gas_response = requests.get(gas_url, gas_params)
                    gas_data = gas_response.json()

                    if gas_data.get("status") == "1" and gas_data.get("result"):
                        # Add gas data as new columns
                        price_df["gas_safe_low"] = float(
                            gas_data["result"]["SafeGasPrice"]
                        )
                        price_df["gas_standard"] = float(
                            gas_data["result"]["ProposeGasPrice"]
                        )
                        price_df["gas_fast"] = float(gas_data["result"]["FastGasPrice"])
                        price_df["gas_base_fee"] = float(
                            gas_data["result"]["suggestBaseFee"]
                        )
                    else:
                        # Add default gas columns if API call fails
                        price_df["gas_safe_low"] = 0
                        price_df["gas_standard"] = 0
                        price_df["gas_fast"] = 0
                        price_df["gas_base_fee"] = 0
                except Exception as e:
                    print(f"Error fetching gas data: {e}")
                    # Add default gas columns
                    price_df["gas_safe_low"] = 0
                    price_df["gas_standard"] = 0
                    price_df["gas_fast"] = 0
                    price_df["gas_base_fee"] = 0
            else:
                print("No Etherscan API key found. Using price data only.")
                price_df["gas_safe_low"] = 0
                price_df["gas_standard"] = 0
                price_df["gas_fast"] = 0
                price_df["gas_base_fee"] = 0

            # 4. Add other required columns
            price_df["quote_asset_volume"] = 0
            price_df["number_of_trades"] = 0
            price_df["taker_buy_base_asset_volume"] = 0
            price_df["taker_buy_quote_asset_volume"] = 0

            print(f"Fetched {len(price_df)} candlesticks with gas data")
            return price_df

        except Exception as e:
            print(f"Error fetching data: {e}")
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
