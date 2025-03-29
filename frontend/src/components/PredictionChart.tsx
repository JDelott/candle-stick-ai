'use client'
import { useEffect, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'

interface Prediction {
  hour: string
  price: number | null
  gas: number | null
}

interface PredictionChartProps {
  type: 'price' | 'gas'
}

export default function PredictionChart({ type }: PredictionChartProps) {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        setLoading(true)
        setError(null)
        console.log('Starting predictions fetch...')
        
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 30000) // 30 second timeout
        
        const response = await fetch('/api/predictions', {
          signal: controller.signal,
          headers: {
            'Content-Type': 'application/json',
          },
        })

        clearTimeout(timeoutId)

        if (!response.ok) {
          console.error('Prediction response not ok:', {
            status: response.status,
            statusText: response.statusText
          })
          throw new Error(`Failed to fetch predictions: ${response.statusText}`)
        }

        const data = await response.json()
        console.log('Prediction data received:', data)

        if (!data.success) {
          throw new Error(data.error || 'Failed to get predictions')
        }

        if (!Array.isArray(data.predictions)) {
          console.error('Invalid predictions format:', data)
          throw new Error('Invalid predictions format received')
        }

        setPredictions(data.predictions)
      } catch (error) {
        console.error('Error in prediction fetch:', error)
        setError(error instanceof Error ? error.message : 'An unknown error occurred')
        if (error instanceof Error && error.name === 'AbortError') {
          setError('Request timed out - predictions are taking too long')
        }
      } finally {
        setLoading(false)
      }
    }

    fetchPredictions()
  }, [])

  if (loading) return (
    <div className="h-full flex flex-col items-center justify-center">
      <div className="mb-2">Loading predictions...</div>
      <div className="text-sm text-gray-500">This may take a few seconds</div>
    </div>
  )
  
  if (error) return (
    <div className="h-full flex flex-col items-center justify-center text-red-500">
      <div className="mb-2">Error: {error}</div>
      <button 
        onClick={() => window.location.reload()}
        className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
      >
        Retry
      </button>
    </div>
  )
  
  if (!predictions.length) return (
    <div className="h-full flex items-center justify-center">
      No predictions available
    </div>
  )

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={predictions} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="hour" 
            tick={{ fontSize: 12 }}
            interval={2}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            domain={['auto', 'auto']}
          />
          <Tooltip 
            formatter={(value: number) => 
              type === 'price' ? `$${value.toFixed(2)}` : `${value.toFixed(1)} GWEI`
            }
          />
          <Legend />
          <Line
            type="monotone"
            dataKey={type}
            stroke={type === 'price' ? '#8884d8' : '#82ca9d'}
            name={type === 'price' ? 'ETH Price (USD)' : 'Gas (GWEI)'}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
