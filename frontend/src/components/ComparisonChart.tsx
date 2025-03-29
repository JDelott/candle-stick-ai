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
  ReferenceArea
} from 'recharts'

interface HistoricalData {
  timestamp: string
  price: number
  gas: number
}

interface ComparisonChartProps {
  type: 'price' | 'gas'
  predictions: Array<{ hour: string; price: number; gas: number }>
}

export default function ComparisonChart({ type, predictions }: ComparisonChartProps) {
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchHistorical = async () => {
      try {
        // Fetch last 7 days of historical data
        const response = await fetch('/api/historical')
        if (!response.ok) throw new Error('Failed to fetch historical data')
        const data = await response.json()
        setHistoricalData(data.historical)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load historical data')
      } finally {
        setLoading(false)
      }
    }

    fetchHistorical()
  }, [])

  if (loading) return <div className="h-full flex items-center justify-center">Loading...</div>
  if (error) return <div className="h-full flex items-center justify-center text-red-500">Error: {error}</div>

  // Combine historical and prediction data
  const combinedData = [
    ...historicalData.map(d => ({
      timestamp: new Date(d.timestamp).toLocaleString(),
      [type]: d[type],
      isHistorical: true
    })),
    ...predictions.map(p => ({
      timestamp: p.hour,
      [type]: p[type],
      isPrediction: true
    }))
  ]

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={combinedData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp" 
            tick={{ fontSize: 12 }}
            interval={Math.floor(combinedData.length / 8)}
            angle={-45}
            textAnchor="end"
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            domain={['auto', 'auto']}
          />
          <Tooltip 
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0].payload
                return (
                  <div className="bg-white p-3 border rounded shadow">
                    <p className="text-sm font-medium">{data.timestamp}</p>
                    <p className="text-sm">
                      {type === 'price' ? `$${payload[0].value}` : `${payload[0].value} GWEI`}
                    </p>
                    <p className="text-xs text-gray-500">
                      {data.isHistorical ? 'Historical' : 'Predicted'}
                    </p>
                  </div>
                )
              }
              return null
            }}
          />
          <Legend />
          <ReferenceArea
            x1={historicalData[historicalData.length - 1]?.timestamp}
            x2={predictions[0]?.hour}
            fill="#f8f9fa"
            fillOpacity={0.3}
            label={{ value: "Prediction Start", position: "insideTop" }}
          />
          <Line
            type="monotone"
            dataKey={type}
            stroke={type === 'price' ? '#8884d8' : '#82ca9d'}
            name={type === 'price' ? 'ETH Price (USD)' : 'Gas (GWEI)'}
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
