const AXIS_LABELS = {
  required_skills: 'Skills',
  domain_fit: 'Domain',
  experience_fit: 'Experience',
  education_fit: 'Education',
  preferred_skills: 'Preferred',
}

const AXIS_TITLES = {
  required_skills: 'Required skills',
  domain_fit: 'Domain / project fit',
  experience_fit: 'Experience fit',
  education_fit: 'Education fit',
  preferred_skills: 'Preferred skills',
}

const SIZE = 360
const CX = SIZE / 2
const CY = 188
const RADIUS = 104
const RINGS = 4
const LABEL_RATIO = 1.32

/**
 * Five-axis radar of the rubric components returned by MatchingService.
 * Written by hand in SVG so the frontend keeps zero charting dependencies.
 */
export default function ScoreRadar({ components = [] }) {
  if (!components.length) return null

  const count = components.length

  const at = (index, ratio) => {
    const angle = (index / count) * Math.PI * 2 - Math.PI / 2
    return [CX + RADIUS * ratio * Math.cos(angle), CY + RADIUS * ratio * Math.sin(angle)]
  }

  const ringPoints = (ratio) =>
    Array.from({ length: count }, (_, index) => at(index, ratio).join(',')).join(' ')

  const dataPoints = components.map((component, index) =>
    at(index, Math.max(0.06, Math.min(1, component.score / 100))),
  )

  const labelAnchor = (x) => {
    if (Math.abs(x - CX) < 8) return 'middle'
    return x > CX ? 'start' : 'end'
  }

  return (
    <div className="radar">
      <svg
        viewBox={`0 0 ${VIEW_W = SIZE} ${VIEW_H = SIZE}`}
        className="radar__svg"
        role="img"
        aria-label="Radar chart of the weighted score breakdown"
      >
        <g className="radar__grid">
          {Array.from({ length: RINGS }, (_, ring) => (
            <polygon key={ring} points={ringPoints((ring + 1) / RINGS)} />
          ))}
          {components.map((component, index) => {
            const [x, y] = at(index, 1)
            return <line key={component.name} x1={CX} y1={CY} x2={x} y2={y} />
          })}
        </g>

        <g className="radar__data">
          <polygon className="radar__area" points={dataPoints.map((p) => p.join(',')).join(' ')} />
          {dataPoints.map((point, index) => (
            <circle
              key={components[index].name}
              className="radar__dot"
              cx={point[0]}
              cy={point[1]}
              r="5"
            />
          ))}
        </g>

        {components.map((component, index) => {
          const [x, y] = at(index, LABEL_RATIO)
          return (
            <g key={`${component.name}-label`}>
              <text className="radar__label" x={x} y={y} textAnchor={labelAnchor(x)}>
                {AXIS_LABELS[component.name] ?? component.name}
              </text>
              <text className="radar__score" x={x} y={y + 15} textAnchor={labelAnchor(x)}>
                {component.score}
                <tspan className="radar__score-unit">/100</tspan>
              </text>
              <title>{AXIS_TITLES[component.name] ?? component.name}</title>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
