import { useState } from 'react'
import ResumeUpload from './components/ResumeUpload.jsx'
import JobDescriptionInput from './components/JobDescriptionInput.jsx'
import MatchResults from './components/MatchResults.jsx'
import DemoBanner from './components/DemoBanner.jsx'
import { analyzeMatch, uploadResume } from './api/client.js'

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

  const canAnalyze =
    Boolean(resume?.extracted_text) && jobDescription.trim().length > 0 && !isAnalyzing

  return (
    <div className="container">
      <header className="app-header">
        <h1>CareerPilot</h1>
        <p>Upload your resume, paste a job description, and see how well you match.</p>
      </header>

      <main>
        <DemoBanner />
        <section className="panel">
          <ResumeUpload resume={resume} error={uploadError} onFileSelected={handleFileSelected} />
          <JobDescriptionInput value={jobDescription} onChange={setJobDescription} />
          <button className="analyze-btn" disabled={!canAnalyze} onClick={handleAnalyze}>
            {isAnalyzing ? 'Analyzing…' : 'Analyze Match'}
          </button>
          {analysisError && <div className="error-banner">{analysisError}</div>}
        </section>

        {result && <MatchResults data={result} />}
      </main>

      <footer className="app-footer">
        Phase 4.1 — Evidence-grounded AI job-fit analysis with multilingual factuality validation.
      </footer>
    </div>
  )
}
