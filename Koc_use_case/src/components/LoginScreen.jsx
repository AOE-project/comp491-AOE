import { useState } from 'react'
import Leaderboard from './Leaderboard.jsx'
import './LoginScreen.css'

export default function LoginScreen({ onSubmit, records }) {
  const [name, setName] = useState('')
  const [showLeaderboard, setShowLeaderboard] = useState(false)

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = name.trim()
    if (trimmed) onSubmit(trimmed)
  }

  return (
    <div className="login-screen">
      <form className="login-card" onSubmit={handleSubmit}>
        <h1>Campus Building Selector</h1>
        <p>Enter your username to continue</p>
        <input
          type="text"
          autoFocus
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Username"
        />
        <button type="submit" className="primary" disabled={!name.trim()}>
          Continue
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => setShowLeaderboard(true)}
        >
          Leaderboards
          {records.length > 0 && (
            <span className="badge">{records.length}</span>
          )}
        </button>
      </form>

      {showLeaderboard && (
        <Leaderboard
          records={records}
          onClose={() => setShowLeaderboard(false)}
        />
      )}
    </div>
  )
}
