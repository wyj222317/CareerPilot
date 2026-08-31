import { useEffect, useState } from 'react'
import { fetchHealth } from '../api/client.js'

export default function DemoBanner() {
  const [isMock, setIsMock] = useState(false)

  useEffect(() => {
    let cancelled = false
    fetchHealth().then((health) => {
      if (!cancelled && health?.llm_provider === 'mock') {
        setIsMock(true)
      }
    })
    return () => {
      cancelled = true
    }
  }, [])

  if (!isMock) return null

  return (
    <div className="demo-banner" role="note">
      <strong>Demo mode</strong>
      {' '}
      — running on deterministic heuristics, no LLM connected. Chinese and
      English resumes/JDs are supported; connect a real provider (DeepSeek,
      OpenAI, Qwen) for production-quality extraction.
    </div>
  )
}
