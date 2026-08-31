function Chips({ skills }) {
  if (!skills?.length) return <p className="muted">None detected</p>
  return (
    <div className="chips">
      {skills.map((skill) => (
        <span key={skill} className="chip">{skill}</span>
      ))}
    </div>
  )
}

function scoreLabel(score) {
  if (score >= 75) return 'Strong match'
  if (score >= 50) return 'Partial match'
  return 'Weak match'
}

function OptimizationPanel({ optimization }) {
  if (!optimization) return null

  return (
    <div className="optimization-panel">
      <h4>Resume Optimization</h4>
      <p className="muted">{optimization.summary}</p>

      {optimization.actions?.length > 0 && (
        <ul className="optimization-list">
          {optimization.actions.map((action, idx) => (
            <li key={idx} className="optimization-item">
              <div className="optimization-meta">
                <span className="optimization-section">{action.target_section}</span>
                <span className="muted">{action.action_type.replace(/_/g, ' ')}</span>
              </div>
              {action.original_text && (
                <div className="optimization-block">
                  <span className="optimization-label">Original</span>
                  <p>{action.original_text}</p>
                </div>
              )}
              {action.optimized_text && (
                <div className="optimization-block">
                  <span className="optimization-label">Suggested</span>
                  <p>{action.optimized_text}</p>
                </div>
              )}
              <div className="optimization-block">
                <span className="optimization-label">Why</span>
                <p>{action.rationale}</p>
              </div>
              {action.target_requirements?.length > 0 && (
                <div className="optimization-block">
                  <span className="optimization-label">Targets</span>
                  <Chips skills={action.target_requirements} />
                </div>
              )}
              {action.evidence_ids?.length > 0 && (
                <div className="optimization-block">
                  <span className="optimization-label">Evidence</span>
                  <div className="evidence-ids">
                    {action.evidence_ids.map((id) => (
                      <span key={id} className="evidence-id">{id}</span>
                    ))}
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
      )}

      {optimization.gap_warnings?.length > 0 && (
        <div className="gap-warnings">
          <h5>Gap Warnings</h5>
          <ul>
            {optimization.gap_warnings.map((warning, idx) => (
              <li key={idx}>
                <strong>{warning.requirement}</strong>
                <span className="muted">{warning.rationale}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {optimization.validation_issues?.length > 0 && (
        <div className="validation-issues">
          <h5>Validation Issues</h5>
          <ul>
            {optimization.validation_issues.map((issue, idx) => (
              <li key={idx}>
                <strong>{issue.field}</strong>
                <span className="muted">{issue.message}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default function MatchResults({ data }) {
  const { candidate, job_requirements: jobRequirements, match, optimization } = data
  const tone = match.overall_score >= 75 ? 'high' : match.overall_score >= 50 ? 'mid' : 'low'

  return (
    <section className="panel results">
      <h2>Match Results</h2>
      <div className="score-row">
        <div className={`score-badge ${tone}`}>{match.overall_score}</div>
        <div className="score-info">
          <h3>{scoreLabel(match.overall_score)}</h3>
          <p>{match.summary}</p>
        </div>
      </div>
      <div className="score-bar">
        <div className={`score-fill ${tone}`} style={{ width: `${match.overall_score}%` }} />
      </div>

      {match.score_breakdown?.length > 0 && (
        <div className="breakdown">
          <h4>Score breakdown</h4>
          <ul className="breakdown-list">
            {match.score_breakdown.map((component) => (
              <li key={component.name} className="breakdown-item">
                <div className="breakdown-head">
                  <span>{component.name.replace(/_/g, ' ')}</span>
                  <span className="muted">
                    {Math.round(component.weight * 100)}% weight · {component.score}/100
                  </span>
                </div>
                <div className="score-bar small">
                  <div
                    className={`score-fill ${tone}`}
                    style={{ width: `${component.score}%` }}
                  />
                </div>
                <span className="muted">{component.detail}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {match.evidence && match.evidence.claims?.length > 0 && (
        <div className="evidence-panel">
          <h4>
            Evidence
            <span className="muted">
              {' '}· {match.evidence.supported_count} supported · {match.evidence.partially_supported_count} partial · {match.evidence.unsupported_count} unsupported
            </span>
          </h4>
          <ul className="evidence-list">
            {match.evidence.claims.map((claim, idx) => (
              <li key={idx} className={`evidence-item ${claim.support_level}`}>
                <div className="evidence-head">
                  <span className="evidence-claim">{claim.claim}</span>
                  <span className={`evidence-badge ${claim.support_level}`}>
                    {claim.support_level.replace(/_/g, ' ')}
                  </span>
                </div>
                {claim.evidence?.length > 0 && (
                  <ul className="evidence-snippets">
                    {claim.evidence.map((ev, evIdx) => (
                      <li key={evIdx}>
                        <span className="muted">{ev.section}</span>
                        <p>{ev.text}</p>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      <OptimizationPanel optimization={optimization} />

      <div className="result-grid">
        <div>
          <h4>Matched skills</h4>
          <Chips skills={match.matched_skills} />
          <h4>Skills detected in your resume</h4>
          <Chips skills={candidate.skills} />
        </div>
        <div>
          <h4>Skill gaps</h4>
          {match.missing_skills.length ? (
            <ul className="gap-list">
              {match.missing_skills.map((gap) => (
                <li key={gap.skill}>
                  <strong>{gap.skill}</strong>
                  <span>{gap.suggestion}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No gaps detected — you cover all listed requirements.</p>
          )}
          <h4>Required by the job</h4>
          <Chips skills={jobRequirements.required_skills} />
        </div>
      </div>

      {match.strengths.length > 0 && (
        <div>
          <h4>Strengths</h4>
          <ul>
            {match.strengths.map((strength) => (
              <li key={strength}>{strength}</li>
            ))}
          </ul>
        </div>
      )}
      {match.recommendations.length > 0 && (
        <div>
          <h4>Recommendations</h4>
          <ol>
            {match.recommendations.map((recommendation) => (
              <li key={recommendation}>{recommendation}</li>
            ))}
          </ol>
        </div>
      )}
    </section>
  )
}
