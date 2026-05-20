import { useEffect, useRef, useState } from 'react'
import { buildings as initialBuildings } from '../data/buildings.js'
import './BuildingEditor.css'

const STORAGE_KEY = 'campus-editor-polygons'

function pointsArrayToString(points) {
  return points.map((p) => `${Math.round(p.x)},${Math.round(p.y)}`).join(' ')
}

function pointsStringToArray(s) {
  return s
    .split(/\s+/)
    .filter(Boolean)
    .map((token) => {
      const [x, y] = token.split(',').map(Number)
      return { x, y }
    })
}

function nameToId(name) {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
  return slug || 'building'
}

function svgPointFromEvent(svgEl, evt) {
  const pt = svgEl.createSVGPoint()
  pt.x = evt.clientX
  pt.y = evt.clientY
  const ctm = svgEl.getScreenCTM()
  if (!ctm) return null
  const t = pt.matrixTransform(ctm.inverse())
  return { x: t.x, y: t.y }
}

function loadStored() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return null
    return parsed.map((b) => ({
      id: b.id,
      name: b.name,
      points: pointsStringToArray(b.points)
    }))
  } catch {
    return null
  }
}

export default function BuildingEditor() {
  const [polygons, setPolygons] = useState(() => {
    const stored = loadStored()
    if (stored) return stored
    return initialBuildings.map((b) => ({
      id: b.id,
      name: b.name,
      points: pointsStringToArray(b.points)
    }))
  })
  const [current, setCurrent] = useState([])
  const [hoverPoint, setHoverPoint] = useState(null)
  const [showExisting, setShowExisting] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const svgRef = useRef(null)

  useEffect(() => {
    const serializable = polygons.map((p) => ({
      id: p.id,
      name: p.name,
      points: pointsArrayToString(p.points)
    }))
    localStorage.setItem(STORAGE_KEY, JSON.stringify(serializable))
  }, [polygons])

  const handleSvgClick = (e) => {
    if (!svgRef.current) return
    const p = svgPointFromEvent(svgRef.current, e)
    if (!p) return
    setCurrent((prev) => [...prev, p])
  }

  const handleSvgMouseMove = (e) => {
    if (!svgRef.current) return
    if (current.length === 0) return
    setHoverPoint(svgPointFromEvent(svgRef.current, e))
  }

  const handleSvgMouseLeave = () => setHoverPoint(null)

  const finishPolygon = () => {
    if (current.length < 3) {
      window.alert('A polygon needs at least 3 points.')
      return
    }
    const name = window.prompt('Building name?')
    if (!name || !name.trim()) return
    const trimmed = name.trim()
    const baseId = nameToId(trimmed)
    let id = baseId
    let i = 2
    const used = new Set(polygons.map((p) => p.id))
    while (used.has(id)) {
      id = `${baseId}-${i++}`
    }
    setPolygons((prev) => [...prev, { id, name: trimmed, points: current }])
    setCurrent([])
    setHoverPoint(null)
  }

  const undoPoint = () => setCurrent((prev) => prev.slice(0, -1))
  const cancelPolygon = () => {
    setCurrent([])
    setHoverPoint(null)
  }

  const renamePolygon = (id) => {
    const target = polygons.find((p) => p.id === id)
    if (!target) return
    const next = window.prompt('New name?', target.name)
    if (!next || !next.trim()) return
    setPolygons((prev) =>
      prev.map((p) => (p.id === id ? { ...p, name: next.trim() } : p))
    )
  }

  const deletePolygon = (id) => {
    setPolygons((prev) => prev.filter((p) => p.id !== id))
  }

  const clearAll = () => {
    if (window.confirm('Delete all buildings? This cannot be undone.')) {
      setPolygons([])
      setCurrent([])
    }
  }

  useEffect(() => {
    const onKey = (e) => {
      const tag = (e.target.tagName || '').toLowerCase()
      if (tag === 'input' || tag === 'textarea') return
      if (e.key === 'Enter') {
        finishPolygon()
      } else if (e.key === 'Escape') {
        cancelPolygon()
      } else if (e.key === 'Backspace' && current.length > 0) {
        e.preventDefault()
        undoPoint()
      } else if ((e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey)) {
        if (current.length > 0) {
          e.preventDefault()
          undoPoint()
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [current, polygons])

  const exportText =
    'export const buildings = [\n' +
    polygons
      .map(
        (p) =>
          `  {\n    id: '${p.id}',\n    name: ${JSON.stringify(p.name)},\n    points: '${pointsArrayToString(p.points)}'\n  }`
      )
      .join(',\n') +
    '\n]\n'

  const copyExport = async () => {
    try {
      await navigator.clipboard.writeText(exportText)
      window.alert(
        'Copied! Paste this into src/data/buildings.js (replace the existing contents).'
      )
    } catch (err) {
      window.alert('Clipboard copy failed: ' + err.message)
    }
  }

  const previewPoints = hoverPoint ? [...current, hoverPoint] : current

  return (
    <div className="editor">
      <aside className="editor-sidebar">
        <h2>Building Editor</h2>
        <p className="hint">
          Click each <strong>corner</strong> of a building on the map. When done,
          press <kbd>Enter</kbd> to name &amp; save it.
        </p>
        <ul className="key-hints">
          <li><kbd>Click</kbd> add point</li>
          <li><kbd>Enter</kbd> finish polygon</li>
          <li><kbd>Backspace</kbd> undo last point</li>
          <li><kbd>Esc</kbd> cancel current polygon</li>
        </ul>

        <div className="editor-controls">
          <button onClick={finishPolygon} disabled={current.length < 3}>
            Finish ({current.length} pts)
          </button>
          <button onClick={undoPoint} disabled={current.length === 0}>
            Undo
          </button>
          <button onClick={cancelPolygon} disabled={current.length === 0}>
            Cancel
          </button>
        </div>

        <div className="toggles">
          <label>
            <input
              type="checkbox"
              checked={showExisting}
              onChange={(e) => setShowExisting(e.target.checked)}
            />
            Show saved buildings
          </label>
          <label>
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
            />
            Show names on map
          </label>
        </div>

        <h3>Buildings ({polygons.length})</h3>
        {polygons.length === 0 ? (
          <p className="empty">No buildings yet. Click corners on the map to add one.</p>
        ) : (
          <ul className="polygon-list">
            {polygons.map((p) => (
              <li key={p.id}>
                <span className="poly-name">{p.name}</span>
                <div className="poly-actions">
                  <button onClick={() => renamePolygon(p.id)} title="Rename">
                    Rename
                  </button>
                  <button
                    className="danger"
                    onClick={() => deletePolygon(p.id)}
                    title="Delete"
                  >
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}

        <div className="editor-export">
          <button className="primary" onClick={copyExport} disabled={polygons.length === 0}>
            Copy buildings.js
          </button>
          <button className="danger" onClick={clearAll} disabled={polygons.length === 0}>
            Clear all
          </button>
        </div>

        <details>
          <summary>Preview output</summary>
          <pre className="export-preview">{exportText}</pre>
        </details>

        <p className="footer-hint">
          Progress is auto-saved in your browser. Open the normal map at{' '}
          <a href="/">/ (homepage)</a>.
        </p>
      </aside>

      <div className="editor-map">
        <div className="map-frame">
          <img
            src="/final_image.png"
            alt="Campus"
            className="editor-img"
            draggable="false"
          />
          <svg
            ref={svgRef}
            className="editor-overlay"
            viewBox="0 0 3636 4608"
            preserveAspectRatio="xMidYMid meet"
            onClick={handleSvgClick}
            onMouseMove={handleSvgMouseMove}
            onMouseLeave={handleSvgMouseLeave}
          >
            {showExisting &&
              polygons.map((p) => (
                <g key={p.id} className="saved-polygon">
                  <polygon
                    points={pointsArrayToString(p.points)}
                    fill="rgba(40,180,80,0.22)"
                    stroke="rgba(20,130,50,0.9)"
                    strokeWidth="6"
                  />
                  {showLabels && p.points.length > 0 && (
                    <text
                      x={p.points[0].x}
                      y={Math.max(40, p.points[0].y - 12)}
                      fontSize="40"
                      fontWeight="600"
                      fill="rgba(20,80,40,0.95)"
                      paintOrder="stroke"
                      stroke="white"
                      strokeWidth="6"
                    >
                      {p.name}
                    </text>
                  )}
                </g>
              ))}

            {current.length > 0 && (
              <g className="current-polygon">
                <polyline
                  points={pointsArrayToString(previewPoints)}
                  fill="rgba(74,144,226,0.25)"
                  stroke="rgba(74,144,226,0.95)"
                  strokeWidth="6"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                {current.map((p, i) => (
                  <circle
                    key={i}
                    cx={p.x}
                    cy={p.y}
                    r="14"
                    fill="#4a90e2"
                    stroke="white"
                    strokeWidth="3"
                  />
                ))}
                {hoverPoint && (
                  <circle
                    cx={hoverPoint.x}
                    cy={hoverPoint.y}
                    r="10"
                    fill="rgba(74,144,226,0.6)"
                    stroke="white"
                    strokeWidth="2"
                  />
                )}
              </g>
            )}
          </svg>
        </div>
      </div>
    </div>
  )
}
