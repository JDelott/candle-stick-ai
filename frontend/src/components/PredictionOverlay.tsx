import { Line, ReferenceLine } from 'recharts'

interface PredictionOverlayProps {
  predictions: Array<{
    hour: string
    price: number
    gas: number
  }>
  type: 'price' | 'gas'
  startTime: string
}

export default function PredictionOverlay({ predictions, type, startTime }: PredictionOverlayProps) {
  const processedPredictions = predictions.map(p => ({
    timestamp: p.hour,
    value: type === 'gas' ? p.gas : p.price
  }))

  return (
    <>
      <ReferenceLine
        x={startTime}
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
      <Line
        type="monotone"
        data={processedPredictions}
        dataKey="value"
        stroke="#ff7300"
        strokeWidth={3}
        dot={false}
        name="Prediction"
        connectNulls
      />
    </>
  )
}
