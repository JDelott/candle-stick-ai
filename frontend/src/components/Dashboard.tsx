'use client'
import { useEffect, useState } from 'react'
import PredictionChart from './PredictionChart'
import ComparisonChart from './ComparisonChart'

interface Prediction {
  hour: string
  price: number
  gas: number
}

export default function Dashboard() {
  const [timeframe, setTimeframe] = useState<'24h' | '48h' | '7d'>('24h')
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())
  const [predictions, setPredictions] = useState<Prediction[]>([])

  // Set up an interval to update the lastUpdated time every minute
  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdated(new Date());
    }, 60000);
    
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const response = await fetch('/api/predictions')
        if (!response.ok) throw new Error('Failed to fetch predictions')
        const data = await response.json()
        if (data.success && data.predictions) {
          setPredictions(data.predictions)
          setLastUpdated(new Date())
        }
      } catch (error) {
        console.error('Error fetching predictions:', error)
      }
    }

    fetchPredictions()
    // Fetch new predictions every 5 minutes
    const interval = setInterval(fetchPredictions, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [timeframe])

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Current Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <StatCard 
          title="Current ETH Price" 
          value="$1,892.45" 
          change="+2.3%"
          isPositive={true}
        />
        <StatCard 
          title="Gas Price (GWEI)" 
          value="32" 
          change="-5.1%"
          isPositive={false}
        />
        <StatCard 
          title="Network Activity" 
          value="High" 
          subtext="Above average transactions"
        />
      </div>

      {/* Controls */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0">
        <div className="flex space-x-2">
          <TimeframeButton 
            timeframe="24h" 
            current={timeframe} 
            onClick={() => setTimeframe('24h')} 
          />
          <TimeframeButton 
            timeframe="48h" 
            current={timeframe} 
            onClick={() => setTimeframe('48h')} 
          />
          <TimeframeButton 
            timeframe="7d" 
            current={timeframe} 
            onClick={() => setTimeframe('7d')} 
          />
        </div>
        <div className="text-sm text-gray-500">
          Last updated: {lastUpdated.toLocaleTimeString()}
        </div>
      </div>

      {/* Prediction Charts */}
      <div className="space-y-6">
        <h2 className="text-xl font-semibold">Next 24 Hours Prediction</h2>
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

      {/* Historical Comparison Charts */}
      <div className="space-y-6">
        <h2 className="text-xl font-semibold">Historical Comparison</h2>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4">Price History & Predictions</h3>
            <div className="h-[400px]">
              <ComparisonChart type="price" predictions={predictions} />
            </div>
          </div>
          <div className="bg-white rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4">Gas History & Predictions</h3>
            <div className="h-[400px]">
              <ComparisonChart type="gas" predictions={predictions} />
            </div>
          </div>
        </div>
      </div>

      {/* Insights */}
      <div className="bg-white rounded-lg shadow p-4">
        <h3 className="text-lg font-semibold mb-4">Market Insights</h3>
        <div className="space-y-3">
          <InsightItem 
            type="positive" 
            text="Price trend suggests bullish momentum over next 24 hours" 
          />
          <InsightItem 
            type="warning" 
            text="High gas prices expected during Asian trading hours" 
          />
          <InsightItem 
            type="neutral" 
            text="Network activity within normal range" 
          />
        </div>
      </div>
    </div>
  )
}

function StatCard({ title, value, change, isPositive, subtext }: {
  title: string
  value: string
  change?: string
  isPositive?: boolean
  subtext?: string
}) {
  return (
    <div className="bg-white rounded-lg shadow p-4 h-full">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <div className="mt-2 flex items-baseline">
        <p className="text-2xl font-semibold text-gray-900">{value}</p>
        {change && (
          <span className={`ml-2 text-sm font-medium ${
            isPositive ? 'text-green-600' : 'text-red-600'
          }`}>
            {change}
          </span>
        )}
      </div>
      {subtext && (
        <p className="mt-1 text-sm text-gray-500">{subtext}</p>
      )}
    </div>
  )
}

function TimeframeButton({ timeframe, current, onClick }: {
  timeframe: '24h' | '48h' | '7d'
  current: string
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
        timeframe === current
          ? 'bg-blue-600 text-white'
          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
      }`}
    >
      {timeframe}
    </button>
  )
}

function InsightItem({ type, text }: { 
  type: 'positive' | 'warning' | 'neutral'
  text: string 
}) {
  const colors = {
    positive: 'text-green-700 bg-green-50',
    warning: 'text-yellow-700 bg-yellow-50',
    neutral: 'text-gray-700 bg-gray-50'
  }

  return (
    <div className={`p-3 rounded-md ${colors[type]}`}>
      {text}
    </div>
  )
}
