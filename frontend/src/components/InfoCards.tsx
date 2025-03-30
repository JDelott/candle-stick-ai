interface MarketConditions {
  sentiment: number
  volatility: number
  volume_trend: number
}

interface InfoCardsProps {
  predictions: Array<{ hour: string; price: number; gas: number }>
  marketConditions: MarketConditions
}

export default function InfoCards({ predictions, marketConditions }: InfoCardsProps) {
  // Add loading check
  if (!marketConditions) {
    return <div className="animate-pulse">Loading market conditions...</div>
  }

  // Add empty state check
  if (predictions.length === 0) {
    return <div>No predictions available</div>
  }

  // Calculate key metrics
  const currentPrice = predictions[0]?.price
  const lastPrice = predictions[predictions.length - 1]?.price
  const priceChange = currentPrice && lastPrice 
    ? ((lastPrice - currentPrice) / currentPrice) * 100 
    : 0

  const currentGas = predictions[0]?.gas
  const maxGas = Math.max(...predictions.map(p => p.gas))
  const minGas = Math.min(...predictions.map(p => p.gas))

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mb-6">
      {/* Price Prediction Card */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold mb-2">Price Forecast</h3>
        <div className="space-y-2">
          <p>Current: ${currentPrice?.toFixed(2)}</p>
          <p>24h Forecast: ${lastPrice?.toFixed(2)}</p>
          <p className={`font-medium ${priceChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            Expected Change: {priceChange.toFixed(2)}%
          </p>
        </div>
      </div>

      {/* Gas Prediction Card */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold mb-2">Gas Forecast</h3>
        <div className="space-y-2">
          <p>Current: {currentGas} GWEI</p>
          <p>24h Range: {minGas} - {maxGas} GWEI</p>
          <p>Best Time: {predictions.find(p => p.gas === minGas)?.hour}</p>
        </div>
      </div>

      {/* Market Conditions Card */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold mb-2">Market Conditions</h3>
        <div className="space-y-2">
          <p>Sentiment: {getSentimentText(marketConditions.sentiment)}</p>
          <p>Volatility: {getVolatilityText(marketConditions.volatility)}</p>
          <p>Volume Trend: {getVolumeTrendText(marketConditions.volume_trend)}</p>
        </div>
      </div>
    </div>
  )
}

function getSentimentText(sentiment: number): string {
  if (sentiment > 0.5) return "Strongly Bullish"
  if (sentiment > 0.2) return "Moderately Bullish"
  if (sentiment > -0.2) return "Neutral"
  if (sentiment > -0.5) return "Moderately Bearish"
  return "Strongly Bearish"
}

function getVolatilityText(volatility: number): string {
  if (volatility > 0.8) return "Very High"
  if (volatility > 0.6) return "High"
  if (volatility > 0.4) return "Moderate"
  if (volatility > 0.2) return "Low"
  return "Very Low"
}

function getVolumeTrendText(trend: number): string {
  if (trend > 500) return "Strongly Increasing"
  if (trend > 200) return "Moderately Increasing"
  if (trend > -200) return "Stable"
  if (trend > -500) return "Moderately Decreasing"
  return "Strongly Decreasing"
}
