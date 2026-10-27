const MAX_LENGTH = 50000

export default function JobDescriptionInput({ value, onChange }) {
  const length = value.length
  const nearLimit = length > MAX_LENGTH * 0.9

  return (
    <div className="jd-section">
      <div className="jd-head">
        <span className="field-label">Job description</span>
        <span className={`jd-counter${nearLimit ? ' is-warning' : ''}`}>
          {length.toLocaleString()} / {MAX_LENGTH.toLocaleString()}
        </span>
      </div>

      <textarea
        id="job-description"
        rows={10}
        placeholder="Paste the full job description here — responsibilities, requirements, and preferred qualifications all help the rubric."
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  )
}
