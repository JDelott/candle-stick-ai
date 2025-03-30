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

  ReferenceLine,

  ReferenceArea
} from 'recharts'

interface HistoricalData {
  timestamp: string
  price: number
  gas: number
}

interface ProcessedData {
  timestamp: string
  value: number
  type: 'historical' | 'prediction'
}

interface ComparisonChartProps {
  type: 'price' | 'gas'
  predictions: Array<{ hour: string; price: number; gas: number }>
}

export default function ComparisonChart({ type, predictions }: ComparisonChartProps) {
  const [historicalData, setHistoricalData] = useState<HistoricalData[]>([])
  const [transitionPoint, setTransitionPoint] = useState<string | null>(null)

  useEffect(() => {
    const fetchHistorical = async () => {
      try {
        const response = await fetch('/api/historical')
        if (!response.ok) throw new Error('Failed to fetch historical data')
        const data = await response.json()
        if (data.historical && data.historical.length > 0) {
          // Process and clean historical data
          const processedData = data.historical
            .filter((item: HistoricalData) => item[type] !== null && item[type] !== undefined)
            .map((item: HistoricalData) => ({
              ...item,
              // Ensure proper number conversion and handle edge cases
              [type]: type === 'gas' 
                ? Math.max(1, parseFloat(item[type].toString())) 
                : parseFloat(item[type].toString())
            }))
            .sort((a: HistoricalData, b: HistoricalData) => 
              new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
            )

          setHistoricalData(processedData)
          setTransitionPoint(processedData[processedData.length - 1].timestamp)
        }
      } catch (err) {
        console.error('Error fetching historical data:', err)
      }
    }

    fetchHistorical()
  }, [type])

  // Create combined dataset
  const historicalSeries: ProcessedData[] = historicalData.map(d => ({
    timestamp: new Date(d.timestamp).toLocaleString(),
    value: d[type],
    type: 'historical'
  }))

  const predictionSeries: ProcessedData[] = predictions.map(p => ({
    timestamp: p.hour,
    value: p[type],
    type: 'prediction'
  }))

  const combinedData = [...historicalSeries, ...predictionSeries]

  // Calculate appropriate Y-axis domain
  const allValues = combinedData.map(d => d.value).filter(v => v !== null) as number[]
  const minValue = Math.min(...allValues)
  const maxValue = Math.max(...allValues)
  const padding = (maxValue - minValue) * 0.1

  const yDomain = type === 'gas' 
    ? [Math.max(0, minValue - padding), maxValue + padding]
    : [minValue - padding, maxValue + padding]

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={combinedData} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          
          {transitionPoint && (
            <>
              {/* Gray area marking predictions */}
              <ReferenceArea
                x1={transitionPoint}
                x2={combinedData[combinedData.length - 1]?.timestamp}
                fill="#f8f9fa"
                fillOpacity={0.3}
              />
              
              {/* Transition line */}
              <ReferenceLine
                x={transitionPoint}
                stroke="#666"
                strokeWidth={2}
                strokeDasharray="3 3"
                label={{
                  value: 'Predictions Start',
                  position: 'top',
                  fill: '#666',
                  fontSize: 12,
                  fontWeight: 'bold'
                }}
              />
            </>
          )}

          <XAxis 
            dataKey="timestamp"
            tick={{ fontSize: 12 }}
            interval={Math.floor(combinedData.length / 8)}
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
              type === 'price' ? `$${value.toFixed(2)}` : `${value.toFixed(1)} GWEI`,
              value === null ? 'N/A' : type === 'price' ? 'Price' : 'Gas'
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
            data={historicalSeries}
            connectNulls
          />
          
          {/* Prediction line */}
          <Line
            type="monotone"
            dataKey="value"
            stroke="#ff7300"
            strokeWidth={3}
            dot={false}
            name="Prediction"
            data={predictionSeries}
            connectNulls
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
