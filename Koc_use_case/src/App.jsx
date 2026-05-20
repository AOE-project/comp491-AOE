import { useEffect, useState } from 'react'
import LoginScreen from './components/LoginScreen.jsx'
import CampusMap from './components/CampusMap.jsx'
import BuildingEditor from './components/BuildingEditor.jsx'

const isEditMode =
  typeof window !== 'undefined' &&
  new URLSearchParams(window.location.search).has('edit')

function sortByScore(records) {
  return [...records].sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
}

export default function App() {
  const [username, setUsername] = useState('')
  const [records, setRecords] = useState([])

  useEffect(() => {
    if (isEditMode) return
    fetch('/api/leaderboard')
      .then((r) => (r.ok ? r.json() : []))
      .then((data) => Array.isArray(data) && setRecords(data))
      .catch(() => setRecords([]))
  }, [])

  const submitScore = async (record) => {
    const res = await fetch('/api/leaderboard', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(record)
    })
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const payload = await res.json()
    const saved = payload.record ?? record

    let updated
    setRecords((prev) => {
      updated = [...prev, saved]
      return updated
    })
    const all = updated ?? [...records, saved]
    const sorted = sortByScore(all)
    const idx = sorted.findIndex(
      (r) => r.username === saved.username && r.timestamp === saved.timestamp
    )
    return {
      record: saved,
      rank: idx === -1 ? sorted.length : idx + 1,
      total: all.length,
      top: sorted.slice(0, 10)
    }
  }

  if (isEditMode) {
    return <BuildingEditor />
  }
  if (!username) {
    return <LoginScreen onSubmit={setUsername} records={records} />
  }
  return (
    <CampusMap
      username={username}
      onLogout={() => setUsername('')}
      onSubmitScore={submitScore}
    />
  )
}
