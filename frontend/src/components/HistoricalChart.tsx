import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import PredictionOverlay from './PredictionOverlay'

interface ProcessedData {
  timestamp: string
  value: number
}

interface HistoricalChartProps {
  type: 'price' | 'gas'
  showPredictions?: boolean
  predictions?: Array<{
    hour: string
    price: number
    gas: number
  }>
}

interface HistoricalDataPoint {
  timestamp: string
  value: number
}

export default function HistoricalChart({ 
  type, 
  showPredictions = false,
  predictions = []
}: HistoricalChartProps) {
  const [data, setData] = useState<ProcessedData[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchHistorical = async () => {
      try {
        const response = await fetch(`/api/historical/${type}`)
        if (!response.ok) throw new Error('Failed to fetch historical data')
        const result = await response.json()
        
        if (result.data) {
          // Process historical data points
          const cleanData = result.data
            .map((item: HistoricalDataPoint): ProcessedData => {
              let value = Number(item.value)
              
              // Handle gas price scaling
              if (type === 'gas') {
                value = Math.max(1, value) // Ensure minimum 1 GWEI
                if (value > 1000) {
                  value = value / 1e9 // Convert Wei to GWEI if needed
                }
              }

              return {
                timestamp: new Date(item.timestamp).toLocaleString(),
                value: value
              }
            })
            .filter((item: ProcessedData) => !isNaN(item.value) && item.value > 0)
            .sort((a: ProcessedData, b: ProcessedData) => 
              new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
            )

          setData(cleanData)
          setError(null)
        }
      } catch (err) {
        console.error('Error fetching historical data:', err)
        setError('Failed to load historical data')
      }
    }

    fetchHistorical()
  }, [type])

  if (error) {
    return <div className="text-red-500 p-4">{error}</div>
  }

  // Calculate appropriate Y-axis domain
  const values = data.map(d => d.value)
  const minValue = Math.min(...values)
  const maxValue = Math.max(...values)
  const padding = (maxValue - minValue) * 0.1

  const yDomain = type === 'gas'
    ? [Math.max(0, minValue - padding), maxValue + padding]
    : ['auto', 'auto']

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart 
          data={data}
          margin={{ top: 10, right: 30, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis 
            dataKey="timestamp"
            tick={{ fontSize: 12 }}
            interval={Math.floor(data.length / 6)}
            angle={-45}
            textAnchor="end"
            height={60}
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            domain={yDomain}
            label={{ 
              value: type === 'price' ? 'Price (USD)' : 'Gas (GWEI)',
              angle: -90,
              position: 'insideLeft',
              dy: 50
            }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              borderRadius: '8px',
              padding: '10px',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
            }}
            formatter={(value: number) => [
              type === 'price' 
                ? `$${value.toFixed(2)}` 
                : `${value.toFixed(1)} GWEI`,
              'Historical'
            ]}
          />
          <Legend />
          
          {/* Historical line */}
          <Line
            type="monotone"
            dataKey="value"
            stroke={type === 'price' ? '#8884d8' : '#82ca9d'}
            strokeWidth={2}
            dot={false}
            name="Historical"
            connectNulls
          />

          {/* Prediction overlay */}
          {showPredictions && predictions.length > 0 && (
            <PredictionOverlay
              type={type}
              predictions={predictions}
              startTime={data[data.length - 1]?.timestamp || ''}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
