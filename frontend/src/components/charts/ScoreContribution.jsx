import { useEffect, useState } from 'react'

const LABELS = {
  required_skills: 'Required skills',
  domain_fit: 'Domain / project fit',
  experience_fit: 'Experience fit',
  education_fit: 'Education fit',
  preferred_skills: 'Preferred skills',
}

const WEAK_THRESHOLD = 60

/**
 * Weighted-contribution bars: bar width encodes the rubric weight, the filled
 * part encodes how much of that weight the candidate actually earned. This is
 * what makes the "which dimension is dragging the score down" question obvious.
 */
export default function ScoreContribution({ components = [] }) {
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const frame = requestAnimationFrame(() => setReady(true))
    return () => cancelAnimationFrame(frame)
  }, [])

  if (!components.length) return null

  const total = components.reduce((sum, component) => sum + component.weight, 0)

  return (
    <ul className="contrib">
      {components.map((component) => {
        const weightPct = total ? (component.weight / total) * 100 : 0
        const earned = Math.max(0, Math.min(100, component.score)) / 100
        const weak = component.score < WEAK_THRESHOLD
        const points = component.weight * component.score

        return (
          <li key={component.name} className={`contrib__item${weak ? ' is-weak' : ''}`}>
            <div className="contrib__head">
              <span className="contrib__name">{LABELS[component.name] ?? component.name}</span>
              <span className="contrib__points">
                {points.toFixed(1)}
                <span className="contrib__points-max"> / {Math.round(weightPct)} pts</span>
              </span>
            </div>

            <div className="contrib__track" style={{ width: `${weightPct}%` }}>
              <div
                className="contrib__fill"
                style={{ transform: `scaleX(${ready ? earned : 0})` }}
              />
            </div>

            <div className="contrib__foot">
              <span className="contrib__score">
                {component.score}
                <span className="contrib__score-unit">/100</span>
              </span>
              <span className="contrib__weight">{Math.round(component.weight * 100)}% weight</span>
            </div>

            <p className="contrib__detail">{component.detail}</p>
          </li>
        )
      })}
    </ul>
  )
}
