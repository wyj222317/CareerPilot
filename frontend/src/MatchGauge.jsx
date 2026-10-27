import { useEffect, useState } from 'react'

const TONES = [
  { max: 50, color: '#dc2626', soft: '#fecaca', label: 'Weak match' },
  { max: 75, color: '#d97706', soft: '#fde68a', label: 'Partial match' },
  { max: Infinity, color: '#16a34a', soft: '#bbf7d0', label: 'Strong match' },
]

export function scoreTone(score) {
  return TONES.find((tone) => score < tone.max) ?? TONES[TONES.length - 1]
}

const WIDTH = 280
const HEIGHT = 186
const CX = WIDTH / 2
const CY = 152
const RADIUS = 106
const STROKE = 20
const ARC_LENGTH = Math.PI * RADIUS
const ARC_PATH = `M ${CX - RADIUS} ${CY} A ${RADIUS} ${RADIUS} 0 0 1 ${CX + RADIUS} ${CY}`

/**
 * Semi-circular gauge for the deterministic overall match score.
 * Pure SVG — no charting dependency, so the Docker/Render build stays untouched.
 */
export default function MatchGauge({ score = 0, summary }) {
  const safe = Math.max(0, Math.min(100, Math.round(score)))
  const tone = scoreTone(safe)
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const frame = requestAnimationFrame(() => setProgress(safe))
    return () => cancelAnimationFrame(frame)
  }, [safe])

  const filled = (progress / 100) * ARC_LENGTH

  return (
    <div className="gauge" style={{ '--tone': tone.color, '--tone-soft': tone.soft }}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Overall match score ${safe} out of 100`}
      >
        <defs>
          <linearGradient id="gauge-fill" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={tone.soft} />
            <stop offset="100%" stopColor={tone.color} />
          </linearGradient>
        </defs>

        <path d={ARC_PATH} fill="none" stroke="#e8edf6" strokeWidth={STROKE} strokeLinecap="round" />

        {[0, 25, 50, 75, 100].map((tick) => {
          const angle = Math.PI - (tick / 100) * Math.PI
          const inner = RADIUS - STROKE / 2 - 4
          const outer = RADIUS + STROKE / 2 + 4
          return (
            <line
              key={tick}
              x1={CX + inner * Math.cos(angle)}
              y1={CY - inner * Math.sin(angle)}
              x2={CX + outer * Math.cos(angle)}
              y2={CY - outer * Math.sin(angle)}
              stroke="#d7dfec"
              strokeWidth="1.5"
            />
          )
        })}

        <path
          className="gauge__value"
          d={ARC_PATH}
          fill="none"
          stroke="url(#gauge-fill)"
          strokeWidth={STROKE}
          strokeLinecap="round"
          strokeDasharray={`${filled} ${ARC_LENGTH}`}
        />

        <text className="gauge__number" x={CX} y={CY - 42} textAnchor="middle">
          {safe}
        </text>
        <text className="gauge__unit" x={CX} y={CY - 18} textAnchor="middle">
          / 100
        </text>
      </svg>

      <div className="gauge__caption">
        <span className="gauge__badge">{tone.label}</span>
        {summary && <p className="gauge__summary">{summary}</p>}
      </div>
    </div>
  )
}
