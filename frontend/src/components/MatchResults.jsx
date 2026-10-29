import MatchGauge from './charts/MatchGauge.jsx'
import ScoreRadar from './charts/ScoreRadar.jsx'
import ScoreContribution from './charts/ScoreContribution.jsx'
import EvidenceBar from './charts/EvidenceBar.jsx'
import ErrorBoundary from './ErrorBoundary.jsx'

function Chips({ skills, variant = 'default' }) {
  if (!skills?.length) return <p className="muted">None detected</p>
  return (
    <div className="chips">
      {skills.map((skill) => (
        <span key={skill} className={`chip chip--${variant}`}>
          {skill}
        </span>
      ))}
    </div>
  )
}

function StatCard({ value, label, hint, tone }) {
  return (
    <div className={`stat${tone ? ` stat--${tone}` : ''}`}>
      <span className="stat__value">{value}</span>
      <span className="stat__label">{label}</span>
      {hint && <span className="stat__hint">{hint}</span>}
    </div>
  )
}

function OptimizationPanel({ optimization }) {
  if (!optimization) return null
  return (
    <section className="panel optimization-panel">
      <header className="panel__head">
        <h3 className="panel__title">Resume optimization</h3>
        <p className="panel__hint">
          Every suggestion below is constrained to resume evidence — nothing is invented.
        </p>
      </header>

      {optimization.summary && <p className="optimization-summary">{optimization.summary}</p>}

      {optimization.actions?.length > 0 && (
        <ul className="optimization-list">
          {optimization.actions.map((action, idx) => (
            <li key={idx} className="optimization-item">
              <div className="optimization-meta">
                <span className="optimization-section">{action.target_section}</span>
                <span className="optimization-tag">{action.action_type.replace(/_/g, ' ')}</span>
              </div>

              {action.original_text && (
                <div className="optimization-block">
                  <span className="optimization-label">Original</span>
                  <p className="optimization-original">{action.original_text}</p>
                </div>
              )}

              {action.optimized_text && (
                <div className="optimization-block">
                  <span className="optimization-label">Suggested</span>
                  <p className="optimization-suggested">{action.optimized_text}</p>
                </div>
              )}

              <div className="optimization-block">
                <span className="optimization-label">Why</span>
                <p>{action.rationale}</p>
              </div>

              {action.target_requirements?.length > 0 && (
                <div className="optimization-block">
                  <span className="optimization-label">Targets</span>
                  <Chips skills={action.target_requirements} variant="target" />
                </div>
              )}

              {action.evidence_ids?.length > 0 && (
                <div className="optimization-block">
                  <span className="optimization-label">Evidence</span>
                  <div className="evidence-ids">
                    {action.evidence_ids.map((id) => (
                      <span key={id} className="evidence-id">
                        {id}
                      </span>
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
          <h4>Gap warnings</h4>
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
          <h4>Validation issues</h4>
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
    </section>
  )
}

export default function MatchResults({ data }) {
  const { candidate, job_requirements: jobRequirements, match, optimization } = data
  const components = match.score_breakdown ?? []
  const evidence = match.evidence
  const evidenceTotal = evidence
    ? (evidence.supported_count ?? 0) +
      (evidence.partially_supported_count ?? 0) +
      (evidence.unsupported_count ?? 0)
    : 0
  const coverage = evidenceTotal
    ? Math.round(((evidence.supported_count ?? 0) / evidenceTotal) * 100)
    : null

  return (
    <div className="results">
      <section className="panel score-hero">
        <MatchGauge score={match.overall_score} summary={match.summary} />

        <div className="score-hero__side">
          <div className="stats">
            <StatCard
              value={match.matched_skills.length}
              label="Matched skills"
              tone="good"
            />
            <StatCard
              value={match.missing_skills.length}
              label="Skill gaps"
              tone={match.missing_skills.length ? 'warn' : 'good'}
            />
            <StatCard
              value={jobRequirements.required_skills.length}
              label="Required by job"
            />
            <StatCard
              value={coverage === null ? '—' : `${coverage}%`}
              label="Evidence-backed"
              hint={evidenceTotal ? `${evidenceTotal} claims checked` : 'no claims retrieved'}
              tone={coverage !== null && coverage >= 70 ? 'good' : 'warn'}
            />
          </div>
        </div>
      </section>

      {components.length > 0 && (
        <div className="chart-grid">
          <section className="panel">
            <header className="panel__head">
              <h3 className="panel__title">Score radar</h3>
              <p className="panel__hint">Five rubric dimensions, each scored 0–100.</p>
            </header>
            <ScoreRadar components={components} />
          </section>

          <section className="panel">
            <header className="panel__head">
              <h3 className="panel__title">Weighted contribution</h3>
              <p className="panel__hint">Bar width = weight. Fill = how much of it you earned.</p>
            </header>
            <ScoreContribution components={components} />
          </section>
        </div>
      )}

      <section className="panel">
        <header className="panel__head">
          <h3 className="panel__title">Skill coverage</h3>
          <p className="panel__hint">What you already have vs. what the posting still asks for.</p>
        </header>

        <div className="coverage">
          <div className="coverage__col">
            <h4 className="coverage__title coverage__title--good">
              Matched skills <span>{match.matched_skills.length}</span>
            </h4>
            <Chips skills={match.matched_skills} variant="matched" />
          </div>

          <div className="coverage__col">
            <h4 className="coverage__title coverage__title--warn">
              Skill gaps <span>{match.missing_skills.length}</span>
            </h4>
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
          </div>

          <div className="coverage__col">
            <h4 className="coverage__title">
              Required by the job <span>{jobRequirements.required_skills.length}</span>
            </h4>
            <Chips skills={jobRequirements.required_skills} variant="required" />
            {jobRequirements.preferred_skills?.length > 0 && (
              <>
                <h4 className="coverage__title coverage__title--sub">Preferred</h4>
                <Chips skills={jobRequirements.preferred_skills} variant="preferred" />
              </>
            )}
          </div>

          <div className="coverage__col">
            <h4 className="coverage__title">
              Detected in your resume <span>{candidate.skills?.length ?? 0}</span>
            </h4>
            <Chips skills={candidate.skills} variant="resume" />
          </div>
        </div>
      </section>

      {evidence?.claims?.length > 0 && (
        <section className="panel evidence-panel">
          <header className="panel__head">
            <h3 className="panel__title">Evidence validation</h3>
            <p className="panel__hint">
              Every claim is traced back to verbatim resume text before it is shown.
            </p>
          </header>

          <EvidenceBar evidence={evidence} />

          <ul className="evidence-list">
            {evidence.claims.map((claim, idx) => (
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
        </section>
      )}

      {(match.strengths?.length > 0 || match.recommendations?.length > 0) && (
        <div className="chart-grid">
          {match.strengths?.length > 0 && (
            <section className="panel">
              <header className="panel__head">
                <h3 className="panel__title">Strengths</h3>
              </header>
              <ul className="bullet-list bullet-list--good">
                {match.strengths.map((strength) => (
                  <li key={strength}>{strength}</li>
                ))}
              </ul>
            </section>
          )}

          {match.recommendations?.length > 0 && (
            <section className="panel">
              <header className="panel__head">
                <h3 className="panel__title">Recommendations</h3>
              </header>
              <ol className="bullet-list bullet-list--action">
                {match.recommendations.map((recommendation) => (
                  <li key={recommendation}>{recommendation}</li>
                ))}
              </ol>
            </section>
          )}
        </div>
      )}

      <OptimizationPanel optimization={optimization} />
    </div>
  )
}
