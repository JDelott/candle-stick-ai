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
} from 'recharts'

interface Prediction {
  hour: string
  price: number
  gas: number
}

interface PredictionResponse {
  predictions: {
    price: number
    gas: number
  }[]
}

export default function PredictionChart() {
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const response = await fetch('/api/predictions')
        if (!response.ok) {
          throw new Error('Failed to fetch predictions')
        }
        
        const data = await response.json() as PredictionResponse
        
        // Format data for chart
        const formattedData: Prediction[] = data.predictions.map((p, index) => ({
          hour: `Hour ${index + 1}`,
          price: p.price,
          gas: p.gas,
        }))
        
        setPredictions(formattedData)
      } catch (error) {
        if (error instanceof Error) {
          setError(error.message)
        } else {
          setError('An unknown error occurred')
        }
      } finally {
        setLoading(false)
      }
    }

    fetchPredictions()
  }, [])

  if (loading) return <div>Loading predictions...</div>
  if (error) return <div>Error: {error}</div>

  return (
    <div className="w-full p-4">
      <h2 className="text-2xl font-bold mb-4">ETH Predictions</h2>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Price Predictions</h3>
          <LineChart width={500} height={300} data={predictions}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="price" 
              stroke="#8884d8" 
              name="ETH Price (USD)" 
            />
          </LineChart>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-2">Gas Predictions</h3>
          <LineChart width={500} height={300} data={predictions}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hour" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="gas" 
              stroke="#82ca9d" 
              name="Gas (GWEI)" 
            />
          </LineChart>
        </div>
      </div>
    </div>
  )
}
