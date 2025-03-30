import { NextResponse } from 'next/server'

export async function GET(
  request: Request,
  context: { params: { type: string } }
) {
  try {
    const type = context.params.type

    if (type !== 'price' && type !== 'gas') {
      return NextResponse.json(
        { success: false, error: 'Invalid type. Must be "price" or "gas"' },
        { status: 400 }
      )
    }

    console.log(`Fetching historical ${type} data from Flask...`)
    const response = await fetch(`http://127.0.0.1:5000/historical/${type}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      cache: 'no-store'
    })
    
    if (!response.ok) {
      const errorText = await response.text()
      console.error(`Flask server error: ${errorText}`)
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`)
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error(`Failed to fetch historical ${context.params.type} data:`, error)
    return NextResponse.json(
      { 
        success: false, 
        error: error instanceof Error ? error.message : 'An unknown error occurred'
      },
      { status: 500 }
    )
  }
}
