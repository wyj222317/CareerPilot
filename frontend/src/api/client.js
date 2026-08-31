const API_BASE = '/api'

async function errorMessage(response, fallback) {
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') return body.detail
    if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
      return `${body.detail[0].msg} (${body.detail[0].loc?.join('.') ?? 'field'})`
    }
  } catch {
    return fallback
  }
  return fallback
}

export async function uploadResume(file) {
  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE}/resumes/upload`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new Error(await errorMessage(response, `Upload failed (${response.status})`))
  }
  return response.json()
}

export async function analyzeMatch(resumeText, jobDescription) {
  const response = await fetch(`${API_BASE}/analysis/match`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text: resumeText, job_description: jobDescription }),
  })
  if (!response.ok) {
    throw new Error(await errorMessage(response, `Analysis failed (${response.status})`))
  }
  return response.json()
}
