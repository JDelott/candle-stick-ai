import { NextResponse } from 'next/server'

interface Prediction {
  hour: string
  price: number | null
  gas: number | null
}

interface FlaskResponse {
  success: boolean
  predictions: Prediction[]
  market_conditions?: {
    sentiment: number
    volatility: number
    volume_trend: number
  }
  error?: string
}

export async function GET() {
  try {
    console.log('Fetching from Flask backend...')
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      cache: 'no-store'
    })

    if (!response.ok) {
      console.error('Flask response:', {
        status: response.status,
        statusText: response.statusText,
        headers: Object.fromEntries(response.headers.entries())
      })
      throw new Error(`Failed to fetch predictions: ${response.statusText}`)
    }

    const data = await response.json() as FlaskResponse
    console.log('Received data from Flask:', data)
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error in predictions route:', error)
    return NextResponse.json(
      { 
        success: false, 
        error: error instanceof Error ? error.message : 'Failed to fetch predictions'
      } as FlaskResponse,
      { status: 500 }
    )
  }
}
