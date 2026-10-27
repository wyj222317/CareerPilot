const SEGMENTS = [
  { key: 'supported', label: 'Supported' },
  { key: 'partially_supported', label: 'Partial' },
  { key: 'unsupported', label: 'Unsupported' },
]

/**
 * Evidence factuality summary: how many generated claims are backed by verbatim
 * resume text, partially backed, or unverifiable.
 */
export default function EvidenceBar({ evidence }) {
  if (!evidence) return null

  const counts = {
    supported: evidence.supported_count ?? 0,
    partially_supported: evidence.partially_supported_count ?? 0,
    unsupported: evidence.unsupported_count ?? 0,
  }
  const total = counts.supported + counts.partially_supported + counts.unsupported
  if (!total) return null

  return (
    <div className="evidence-bar">
      <div className="evidence-bar__track">
        {SEGMENTS.map((segment) => (
          <div
            key={segment.key}
            className={`evidence-bar__seg is-${segment.key}`}
            style={{ width: `${(counts[segment.key] / total) * 100}%` }}
            title={`${segment.label}: ${counts[segment.key]}`}
          />
        ))}
      </div>

      <ul className="evidence-bar__legend">
        {SEGMENTS.map((segment) => (
          <li key={segment.key}>
            <span className={`evidence-bar__dot is-${segment.key}`} />
            {segment.label}
            <strong>{counts[segment.key]}</strong>
          </li>
        ))}
      </ul>
    </div>
  )
}
