'use client'
import { useEffect, useState } from 'react'
import PredictionChart from './PredictionChart'
import InfoCards from './InfoCards'
import HistoricalChart from './HistoricalChart'
import Chat from './Chat'

interface Prediction {
  hour: string
  price: number | null
  gas: number | null
}

interface MarketConditions {
  sentiment: number
  volatility: number
  volume_trend: number
}

// Helper function to filter out null values
const filterValidPredictions = (predictions: Prediction[]) => {
  return predictions.filter(
    (p): p is { hour: string; price: number; gas: number } => 
      p.price !== null && p.gas !== null
  )
}

export default function Dashboard() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [marketConditions, setMarketConditions] = useState<MarketConditions>({
    sentiment: 0,
    volatility: 0,
    volume_trend: 0
  })
  const [showPredictions, setShowPredictions] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        // Fetch predictions
        const predResponse = await fetch('/api/predictions')
        if (!predResponse.ok) throw new Error('Failed to fetch predictions')
        const predData = await predResponse.json()
        
        if (predData.success && predData.predictions) {
          setPredictions(predData.predictions)
        }

        // Fetch market conditions
        const marketResponse = await fetch('/api/market-conditions')
        if (!marketResponse.ok) throw new Error('Failed to fetch market conditions')
        const marketData = await marketResponse.json()
        
        if (marketData.success && marketData.conditions) {
          setMarketConditions(marketData.conditions)
        }

        setError(null)
      } catch (err) {
        console.error('Error fetching data:', err)
        setError(err instanceof Error ? err.message : 'An error occurred')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
    // Set up polling every 5 minutes
    const interval = setInterval(fetchData, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  const validPredictions = filterValidPredictions(predictions)

  if (loading) {
    return <div className="flex justify-center items-center min-h-screen">
      <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
    </div>
  }

  if (error) {
    return <div className="text-red-500 p-4 text-center">{error}</div>
  }

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Info Cards */}
        <InfoCards 
          predictions={validPredictions} 
          marketConditions={marketConditions} 
        />

        {/* Prediction Charts */}
        <div className="space-y-6">
          <h2 className="text-xl font-semibold">Next 24 Hours</h2>
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-lg font-semibold mb-4">Price Predictions</h3>
              <div className="h-[300px]">
                <PredictionChart type="price" />
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-lg font-semibold mb-4">Gas Predictions</h3>
              <div className="h-[300px]">
                <PredictionChart type="gas" />
              </div>
            </div>
          </div>
        </div>

        {/* Historical Data */}
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-semibold">Historical Data</h2>
            <button
              onClick={() => setShowPredictions(!showPredictions)}
              className={`px-4 py-2 rounded ${
                showPredictions 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-200 text-gray-700'
              }`}
            >
              {showPredictions ? 'Hide Predictions' : 'Show Predictions'}
            </button>
          </div>
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-lg font-semibold mb-4">ETH Price History</h3>
              <div className="h-[400px]">
                <HistoricalChart 
                  type="price"
                  showPredictions={showPredictions}
                  predictions={validPredictions}
                />
              </div>
            </div>
            <div className="bg-white rounded-lg shadow p-4">
              <h3 className="text-lg font-semibold mb-4">Gas Price History</h3>
              <div className="h-[400px]">
                <HistoricalChart 
                  type="gas"
                  showPredictions={showPredictions}
                  predictions={validPredictions}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Add Chat component */}
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-lg font-semibold mb-4">Chat with AI Assistant</h3>
          <Chat />
        </div>
      </div>
    </div>
  )
}
