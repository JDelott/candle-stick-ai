from typing import Dict
import anthropic
from dotenv import load_dotenv
import os
import pandas as pd
from ..data_loader import DataLoader
from ..predict import Predictor
import logging

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.data_loader = DataLoader()
        self.predictor = Predictor()

    def get_response(self, question: str, market_data: Dict) -> Dict:
        """Generate a response using Claude with access to model's knowledge"""
        try:
            df = self.data_loader.fetch_combined_data()

            # Get model's predictions
            gas_predictions, _ = self.predictor._predict_gas(
                df, n=6, conditions=market_data
            )
            price_predictions, _ = self.predictor._predict_price(
                df, n=6, conditions=market_data
            )

            # Simple context that gives Claude access to the model's knowledge
            context = f"""
            You are an AI assistant with direct access to a trained Ethereum prediction model.
            The model has learned patterns from historical ETH price and gas data, including:
            - Price-gas correlations
            - Market cycles and trends
            - Network congestion patterns
            - Trading volume impacts
            - Time-of-day effects
            
            Current Market:
            ETH: ${market_data['eth_price']:.2f} ({market_data['price_trend']})
            Gas: {market_data['gas_price']:.1f} GWEI ({market_data['gas_trend']})
            
            Model's Latest Predictions:
            Price: {[f'${p:.2f}' for p in price_predictions]}
            Gas: {[f'{p:.1f}' for p in gas_predictions]} GWEI

            Answer questions naturally about the model's predictions, training data, 
            and why it makes specific forecasts. Use the model's learned patterns 
            to explain its reasoning.
            """

            message = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                temperature=0.7,
                system=context,
                messages=[{"role": "user", "content": question}],
            )

            return {"success": True, "response": message.content[0].text}

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return {"success": False, "error": str(e)}
