export default function JobDescriptionInput({ value, onChange }) {
  return (
    <div className="jd-section">
      <label className="field-label" htmlFor="job-description">Job Description</label>
      <textarea
        id="job-description"
        rows={10}
        placeholder="Paste the full job description here…"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </div>
  )
}
