import { useState } from 'react'

export default function ResumeUpload({ resume, error, onFileSelected }) {
  const [uploading, setUploading] = useState(false)

  async function handleChange(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setUploading(true)
    await onFileSelected(file)
    setUploading(false)
  }

  return (
    <div className="upload-section">
      <label className="field-label" htmlFor="resume-file">Resume (PDF)</label>
      <div className="upload-row">
        <input
          id="resume-file"
          type="file"
          accept="application/pdf,.pdf"
          onChange={handleChange}
          disabled={uploading}
        />
        {uploading && <span className="muted">Extracting text…</span>}
      </div>
      {error && <div className="error-banner">{error}</div>}
      {resume && (
        <div className="upload-summary">
          <strong>{resume.filename}</strong> — {resume.page_count} page(s),{' '}
          {resume.character_count.toLocaleString()} characters extracted
          <details>
            <summary>View extracted text</summary>
            <pre>{resume.extracted_text}</pre>
          </details>
        </div>
      )}
    </div>
  )
}
