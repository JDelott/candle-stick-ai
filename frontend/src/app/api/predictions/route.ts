import { NextResponse } from 'next/server'

interface PredictionResponse {
  success: boolean
  predictions?: {
    price: number
    gas: number
  }[]
  error?: string
}

export async function GET() {
  try {
    console.log('Fetching predictions from backend...')
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      cache: 'no-store'  // Disable caching
    })
    
    if (!response.ok) {
      console.error('Backend response not OK:', response.status, response.statusText)
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json() as PredictionResponse
    console.log('Received data:', data)
    
    if (!data.success) {
      throw new Error(data.error || 'Failed to get predictions')
    }
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('Failed to fetch predictions:', error instanceof Error ? error.message : 'Unknown error')
    return NextResponse.json(
      { 
        success: false, 
        error: error instanceof Error ? error.message : 'An unknown error occurred'
      } satisfies PredictionResponse,
      { status: 500 }
    )
  }
}
