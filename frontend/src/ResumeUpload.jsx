import { useRef, useState } from 'react'

export default function ResumeUpload({ resume, error, onFileSelected }) {
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  async function handleFile(file) {
    if (!file) return
    setUploading(true)
    try {
      await onFileSelected(file)
    } finally {
      setUploading(false)
    }
  }

  function handleChange(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    handleFile(file)
  }

  function handleDrop(event) {
    event.preventDefault()
    setDragging(false)
    handleFile(event.dataTransfer.files?.[0])
  }

  const isPdf = (file) => file && (file.type === 'application/pdf' || /\.pdf$/i.test(file.name))

  return (
    <div className="upload-section">
      <span className="field-label">Resume</span>

      <div
        className={`upload-drop${dragging ? ' is-dragging' : ''}${uploading ? ' is-busy' : ''}`}
        onClick={() => !uploading && inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click()
        }}
      >
        <span className="upload-drop__icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 16V4" />
            <path d="m7 9 5-5 5 5" />
            <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
          </svg>
        </span>

        <span className="upload-drop__title">
          {uploading ? 'Extracting text…' : 'Drop your resume here'}
        </span>
        <span className="upload-drop__hint">PDF, up to 10 MB — or click to browse</span>

        <input
          ref={inputRef}
          id="resume-file"
          className="upload-drop__input"
          type="file"
          accept="application/pdf,.pdf"
          onChange={handleChange}
          disabled={uploading}
        />
      </div>

      {error && <div className="error-banner">{error}</div>}

      {resume && (
        <div className="upload-meta">
          <span className="upload-meta__icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 6 9 17l-5-5" />
            </svg>
          </span>
          <div className="upload-meta__body">
            <strong>{resume.filename}</strong>
            <span className="muted">
              {resume.page_count} page{resume.page_count === 1 ? '' : 's'} ·{' '}
              {resume.character_count.toLocaleString()} characters extracted
            </span>
          </div>
          <details className="upload-meta__details">
            <summary>View extracted text</summary>
            <pre>{resume.extracted_text}</pre>
          </details>
        </div>
      )}
    </div>
  )
}
