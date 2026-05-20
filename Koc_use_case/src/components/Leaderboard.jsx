import './Leaderboard.css'

function formatTimestamp(iso) {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString()
}

export default function Leaderboard({ records, onClose }) {
  const sorted = [...records].sort((a, b) => (b.score ?? 0) - (a.score ?? 0))

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal leaderboard-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="leaderboard-head">
          <h2>Leaderboards</h2>
          <button
            type="button"
            className="close-btn"
            onClick={onClose}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {sorted.length === 0 ? (
          <p className="empty-state">
            No records yet. Sign in and submit a calculation to populate the
            leaderboard.
          </p>
        ) : (
          <div className="leaderboard-table-wrap">
            <table className="leaderboard-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>User</th>
                  <th>Buildings</th>
                  <th>Score</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((r, i) => (
                  <tr key={i}>
                    <td className="rank">{i + 1}</td>
                    <td className="user">{r.username}</td>
                    <td className="buildings">{r.buildings}</td>
                    <td className="score">{r.score}</td>
                    <td className="date">{formatTimestamp(r.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
