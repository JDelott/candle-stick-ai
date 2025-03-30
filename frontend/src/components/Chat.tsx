import { useState } from 'react'

export default function Chat() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([])
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim()) return

    setLoading(true)
    setMessages(prev => [...prev, { role: 'user', content: input }])
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: input })
      })
      
      const data = await response.json()
      
      if (data.success) {
        setMessages(prev => [...prev, { role: 'assistant', content: data.response }])
      } else {
        throw new Error(data.error)
      }
    } catch (error) {
      console.error('Chat error:', error)
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error processing your request.' 
      }])
    } finally {
      setLoading(false)
      setInput('')
    }
  }

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="h-[400px] overflow-y-auto mb-4 space-y-4">
        {messages.map((msg, idx) => (
          <div 
            key={idx} 
            className={`p-3 rounded-lg ${
              msg.role === 'user' 
                ? 'bg-blue-100 ml-12' 
                : 'bg-gray-100 mr-12'
            }`}
          >
            {msg.content}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about ETH price or gas..."
          className="flex-1 p-2 border rounded"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400"
        >
          {loading ? 'Thinking...' : 'Send'}
        </button>
      </form>
    </div>
  )
}
