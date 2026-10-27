import { useState } from 'react'
import ResumeUpload from './components/ResumeUpload.jsx'
import JobDescriptionInput from './components/JobDescriptionInput.jsx'
import MatchResults from './components/MatchResults.jsx'
import DemoBanner from './components/DemoBanner.jsx'
import { analyzeMatch, uploadResume } from './api/client.js'

const STEPS = ['Upload resume', 'Paste job description', 'Analyze match']

export default function App() {
  const [resume, setResume] = useState(null)
  const [uploadError, setUploadError] = useState(null)
  const [jobDescription, setJobDescription] = useState('')
  const [result, setResult] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState(null)

  async function handleFileSelected(file) {
    setUploadError(null)
    setResume(null)
    setResult(null)
    if (!file) return
    try {
      setResume(await uploadResume(file))
    } catch (err) {
      setUploadError(err.message)
    }
  }

  async function handleAnalyze() {
    setAnalysisError(null)
    setIsAnalyzing(true)
    try {
      setResult(await analyzeMatch(resume.extracted_text, jobDescription))
    } catch (err) {
      setResult(null)
      setAnalysisError(err.message)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const hasResume = Boolean(resume?.extracted_text)
  const hasJd = jobDescription.trim().length > 0
  const canAnalyze = hasResume && hasJd && !isAnalyzing
  const activeStep = result ? 3 : hasResume && hasJd ? 2 : hasResume ? 1 : 0

  return (
    <div className="container">
      <header className="app-header">
        <div className="app-header__brand">
          <span className="app-header__logo" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3 3 8l9 5 9-5-9-5Z" />
              <path d="M3 16l9 5 9-5" />
              <path d="M3 12l9 5 9-5" />
            </svg>
          </span>
          <div>
            <h1 className="app-header__title">CareerPilot</h1>
            <p className="app-header__subtitle">
              Evidence-grounded job-fit analysis and resume optimization.
            </p>
          </div>
        </div>

        <ol className="stepper">
          {STEPS.map((step, index) => {
            const state = index < activeStep ? 'is-done' : index === activeStep ? 'is-active' : ''
            return (
              <li key={step} className={`stepper__item ${state}`}>
                <span className="stepper__index">{index + 1}</span>
                <span className="stepper__label">{step}</span>
              </li>
            )
          })}
        </ol>
      </header>

      <main>
        <DemoBanner />

        <section className="panel">
          <div className="form-grid">
            <ResumeUpload resume={resume} error={uploadError} onFileSelected={handleFileSelected} />
            <JobDescriptionInput value={jobDescription} onChange={setJobDescription} />
          </div>

          <button
            className={`analyze-btn${isAnalyzing ? ' is-loading' : ''}`}
            disabled={!canAnalyze}
            onClick={handleAnalyze}
          >
            {isAnalyzing ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Analyzing your match…
              </>
            ) : (
              'Analyze match'
            )}
          </button>

          {!canAnalyze && !isAnalyzing && (
            <p className="analyze-hint">
              {!hasResume
                ? 'Upload a PDF resume to continue.'
                : !hasJd
                  ? 'Paste the job description to continue.'
                  : ''}
            </p>
          )}

          {analysisError && <div className="error-banner">{analysisError}</div>}
        </section>

        {isAnalyzing && !result && (
          <section className="panel results-skeleton" aria-hidden="true">
            <div className="skeleton skeleton--hero" />
            <div className="skeleton-grid">
              <div className="skeleton skeleton--block" />
              <div className="skeleton skeleton--block" />
            </div>
          </section>
        )}

        {result && <MatchResults data={result} />}
      </main>

      <footer className="app-footer">
        Phase 4.1 — Evidence-grounded AI job-fit analysis with multilingual factuality validation.
      </footer>
    </div>
  )
}
