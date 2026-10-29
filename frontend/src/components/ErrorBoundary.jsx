import { Component } from 'react'

export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('[CareerPilot] render failed:', error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            padding: '16px 18px',
            border: '1px solid #fecaca',
            borderRadius: '12px',
            background: '#fef2f2',
            color: '#b91c1c',
            fontSize: '13px',
            lineHeight: 1.6,
          }}
        >
          <strong>This section failed to render.</strong>
          <div style={{ marginTop: 4, color: '#7f1d1d', opacity: 0.85 }}>
            {this.state.error.message}
          </div>
          <button
            type="button"
            onClick={() => this.setState({ error: null })}
            style={{
              marginTop: 10,
              padding: '6px 14px',
              border: '1px solid #fca5a5',
              borderRadius: '8px',
              background: '#fff',
              color: '#b91c1c',
              fontSize: '12px',
              cursor: 'pointer',
            }}
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
