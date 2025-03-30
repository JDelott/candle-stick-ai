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
  ResponsiveContainer,
  Area,
  ReferenceLine
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
        <LineChart data={predictions} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
          <defs>
            <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={type === 'price' ? '#8884d8' : '#82ca9d'} stopOpacity={0.1}/>
              <stop offset="95%" stopColor={type === 'price' ? '#8884d8' : '#82ca9d'} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="hour"
            tick={{ fontSize: 12 }}
            interval={2}
            angle={-15}
            dy={10}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            domain={['auto', 'auto']}
            label={{ 
              value: type === 'price' ? 'Price (USD)' : 'Gas (GWEI)', 
              angle: -90, 
              position: 'insideLeft',
              dy: 50
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.9)',
              borderRadius: '8px',
              padding: '10px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
            formatter={(value: number) => 
              type === 'price' ? `$${value.toFixed(2)}` : `${value.toFixed(1)} GWEI`
            }
          />
          <Legend 
            verticalAlign="top"
            height={36}
          />
          <Area
            type="monotone"
            dataKey={type}
            stroke={type === 'price' ? '#8884d8' : '#82ca9d'}
            fillOpacity={1}
            fill="url(#colorValue)"
          />
          <Line
            type="monotone"
            dataKey={type}
            stroke={type === 'price' ? '#8884d8' : '#82ca9d'}
            strokeWidth={2}
            dot={false}
            name={type === 'price' ? 'ETH Price' : 'Gas Price'}
          />
          <ReferenceLine
            y={type === 'price' ? predictions[0]?.price ?? undefined : predictions[0]?.gas ?? undefined}
            stroke="#666"
            strokeDasharray="3 3"
            label={{ 
              value: 'Current',
              position: 'insideTopLeft',
              fill: '#666'
            }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
