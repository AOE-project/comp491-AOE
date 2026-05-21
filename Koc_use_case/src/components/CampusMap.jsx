import { useEffect, useRef, useState } from 'react'
import { buildings } from '../data/buildings.js'
import demand from '../data/demand.json'
import fixedCostDaily from '../data/fixed_cost_daily.json'
import capacity from '../data/capacity.json'
import Fireworks from './Fireworks.jsx'
import './CampusMap.css'

const MAX_SELECTED = 3
const MIN_ZOOM = 0.5
const MAX_ZOOM = 2
const IMG_NATURAL_W = 3636
const IMG_NATURAL_H = 4608
const FIREWORKS_RANK_THRESHOLD = 3

let visitProb = null

const MODEL_DATA = {
  demand,
  fixed_cost_daily: fixedCostDaily,
  capacity,
  visit_prob: null,
  rev_per_customer: 150
}

async function loadVisitProb() {
  if (!visitProb) {
    const data = await import('../data/visit_prob.json')
    visitProb = data.default
    MODEL_DATA.visit_prob = visitProb
  }
  return visitProb
}

let costCache = {}

async function calculateCost(selectedIds) {
  if (selectedIds.length === 0) return 0

  const cacheKey = selectedIds.sort().join(',')
  if (costCache[cacheKey] !== undefined) {
    return costCache[cacheKey]
  }

  const selectedNames = selectedIds
    .map(id => buildings.find(b => b.id === id)?.name)
    .filter(Boolean)

  try {
    const response = await fetch('http://127.0.0.1:5000/api/compute-score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ shops: selectedNames })
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`)
    }

    const data = await response.json()
    const result = data.score
    costCache[cacheKey] = result
    return result
  } catch (err) {
    console.error('Error calling score API:', err)
    return null
  }
}

function ordinal(n) {
  if (!Number.isFinite(n)) return ''
  const s = ['th', 'st', 'nd', 'rd']
  const v = n % 100
  return n + (s[(v - 20) % 10] || s[v] || s[0])
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v))
}

export default function CampusMap({ username, onLogout, onSubmitScore }) {
  const [selected, setSelected] = useState([])
  const [hoveredId, setHoveredId] = useState(null)
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 })
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const [zoom, setZoom] = useState(1)
  const [baseSize, setBaseSize] = useState({ w: 0, h: 0 })
  const viewportRef = useRef(null)

  useEffect(() => {
    loadVisitProb().catch(err => console.error('Failed to load visit probability data:', err))
  }, [])

  useEffect(() => {
    const compute = () => {
      const vp = viewportRef.current
      if (!vp) return
      const rect = vp.getBoundingClientRect()
      const aspect = IMG_NATURAL_W / IMG_NATURAL_H
      const fitByWidth = rect.width / rect.height < aspect
      const w = fitByWidth ? rect.width : rect.height * aspect
      const h = fitByWidth ? rect.width / aspect : rect.height
      setBaseSize({ w, h })
    }
    compute()
    const ro = new ResizeObserver(compute)
    if (viewportRef.current) ro.observe(viewportRef.current)
    window.addEventListener('resize', compute)
    return () => {
      ro.disconnect()
      window.removeEventListener('resize', compute)
    }
  }, [])

  useEffect(() => {
    const el = viewportRef.current
    if (!el) return
    const onWheel = (e) => {
      if (!e.ctrlKey) return
      e.preventDefault()
      const rect = el.getBoundingClientRect()
      const cursorX = e.clientX - rect.left
      const cursorY = e.clientY - rect.top
      setZoom((prevZoom) => {
        const delta = e.deltaY > 0 ? 0.9 : 1.1
        const newZoom = clamp(prevZoom * delta, MIN_ZOOM, MAX_ZOOM)
        if (newZoom === prevZoom) return prevZoom
        const factor = newZoom / prevZoom
        const sx = el.scrollLeft + cursorX
        const sy = el.scrollTop + cursorY
        requestAnimationFrame(() => {
          el.scrollLeft = sx * factor - cursorX
          el.scrollTop = sy * factor - cursorY
        })
        return newZoom
      })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [])

  const stepZoom = (factor) => {
    const el = viewportRef.current
    setZoom((prevZoom) => {
      const newZoom = clamp(prevZoom * factor, MIN_ZOOM, MAX_ZOOM)
      if (newZoom === prevZoom || !el) return newZoom
      const rect = el.getBoundingClientRect()
      const cursorX = rect.width / 2
      const cursorY = rect.height / 2
      const f = newZoom / prevZoom
      const sx = el.scrollLeft + cursorX
      const sy = el.scrollTop + cursorY
      requestAnimationFrame(() => {
        el.scrollLeft = sx * f - cursorX
        el.scrollTop = sy * f - cursorY
      })
      return newZoom
    })
  }

  const resetZoom = () => {
    setZoom(1)
    const el = viewportRef.current
    if (el) {
      requestAnimationFrame(() => {
        el.scrollLeft = 0
        el.scrollTop = 0
      })
    }
  }

  const isSelected = (id) => selected.includes(id)
  const limitReached = selected.length >= MAX_SELECTED
  const canCalculate = selected.length === MAX_SELECTED && !submitting

  const handleClick = (id) => {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id)
      if (prev.length >= MAX_SELECTED) return prev
      return [...prev, id]
    })
  }

  const handleMouseMove = (e) => {
    setTooltipPos({ x: e.clientX, y: e.clientY })
  }

  const handleCalculate = async () => {
    if (!canCalculate) return
    setSubmitting(true)
    setError(null)
    const buildingNames = selected
      .map((id) => buildings.find((b) => b.id === id)?.name)
      .filter(Boolean)
    const score = await calculateCost(selected)
    if (score === null) {
      setError('Failed to compute score')
      setSubmitting(false)
      return
    }
    const record = {
      username,
      buildings: buildingNames.join(', '),
      score,
      timestamp: new Date().toISOString()
    }
    try {
      const { rank, total, top, record: saved } = await onSubmitScore(record)
      setResult({
        score,
        buildings: buildingNames,
        rank,
        total,
        top: top ?? [],
        recordTimestamp: saved?.timestamp ?? record.timestamp
      })
    } catch (err) {
      setError('Failed to save record: ' + err.message)
    } finally {
      setSubmitting(false)
    }
  }

  const handleResultDone = () => {
    setResult(null)
    setSelected([])
    onLogout()
  }

  const hoveredBuilding = buildings.find((b) => b.id === hoveredId)
  const contentW = baseSize.w * zoom
  const contentH = baseSize.h * zoom

  return (
    <div className="campus-map">
      <header className="topbar">
        <div className="welcome">
          Welcome, <strong>{username}</strong>
        </div>
        <div className="status">
          <span className="counter">
            Selected <strong>{selected.length}/{MAX_SELECTED}</strong>
          </span>
          {selected.length > 0 && (
            <ul className="selected-list">
              {selected.map((id) => {
                const b = buildings.find((bb) => bb.id === id)
                return (
                  <li key={id}>
                    {b?.name}
                    <button
                      type="button"
                      className="remove-btn"
                      onClick={() => handleClick(id)}
                      aria-label={`Remove ${b?.name}`}
                    >
                      ×
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
          <button type="button" className="logout-btn" onClick={onLogout}>
            Sign out
          </button>
        </div>
      </header>

      <div
        className="map-viewport"
        ref={viewportRef}
        onMouseMove={handleMouseMove}
      >
        {baseSize.w > 0 && (
          <div
            className="map-content"
            style={{ width: contentW, height: contentH }}
          >
            <img
              src="/final_image.png"
              alt="Campus Map"
              className="campus-img"
              draggable="false"
            />
            <svg
              className="overlay"
              viewBox={`0 0 ${IMG_NATURAL_W} ${IMG_NATURAL_H}`}
              preserveAspectRatio="xMidYMid meet"
            >
              {buildings.map((b) => {
                const sel = isSelected(b.id)
                const cls = [
                  'building',
                  sel && 'selected',
                  !sel && limitReached && 'disabled'
                ]
                  .filter(Boolean)
                  .join(' ')
                return (
                  <polygon
                    key={b.id}
                    points={b.points}
                    className={cls}
                    onMouseEnter={() => setHoveredId(b.id)}
                    onMouseLeave={() => setHoveredId(null)}
                    onClick={() => handleClick(b.id)}
                  />
                )
              })}
            </svg>
          </div>
        )}
      </div>

      <div className="zoom-controls">
        <button
          type="button"
          onClick={() => stepZoom(1.25)}
          disabled={zoom >= MAX_ZOOM - 1e-6}
          title="Zoom in (Ctrl + scroll)"
        >
          +
        </button>
        <button
          type="button"
          className="zoom-reset"
          onClick={resetZoom}
          title="Reset zoom"
        >
          {Math.round(zoom * 100)}%
        </button>
        <button
          type="button"
          onClick={() => stepZoom(0.8)}
          disabled={zoom <= MIN_ZOOM + 1e-6}
          title="Zoom out (Ctrl + scroll)"
        >
          −
        </button>
      </div>

      {hoveredBuilding && (
        <div
          className="tooltip"
          style={{ left: tooltipPos.x + 14, top: tooltipPos.y + 14 }}
        >
          {hoveredBuilding.name}
          {!isSelected(hoveredBuilding.id) && limitReached && (
            <span className="tooltip-note"> (limit reached)</span>
          )}
        </div>
      )}

      <button
        type="button"
        className="calculate-btn"
        onClick={handleCalculate}
        disabled={!canCalculate}
        title={
          canCalculate
            ? 'Calculate cost for the selected buildings'
            : `Select ${MAX_SELECTED - selected.length} more building${
                MAX_SELECTED - selected.length === 1 ? '' : 's'
              }`
        }
      >
        {submitting ? 'Calculating…' : 'Calculate Cost'}
      </button>

      {error && (
        <div className="error-toast" onClick={() => setError(null)}>
          {error}
        </div>
      )}

      {result && (
        <div className="modal-backdrop">
          {result.rank <= FIREWORKS_RANK_THRESHOLD && <Fireworks />}
          <div
            className="modal result-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <h2>Cost calculated</h2>
            <div className="result-score">{result.score}</div>
            <p className="result-meta">
              for <strong>{username}</strong>
            </p>

            <div className="rank-block">
              <div className="rank-label">You ranked</div>
              <div className="rank-value">{ordinal(result.rank)}</div>
              <div className="rank-of">of {result.total}</div>
            </div>

            {result.top.length > 0 && (
              <div className="top-list">
                <div className="top-list-title">Top 10 All-Time</div>
                <ol className="top-list-items">
                  {result.top.map((r, i) => {
                    const isMe =
                      r.username === username &&
                      r.timestamp === result.recordTimestamp
                    return (
                      <li
                        key={`${r.username}-${r.timestamp}-${i}`}
                        className={isMe ? 'top-row me' : 'top-row'}
                      >
                        <span className="top-rank">{i + 1}</span>
                        <span className="top-name">{r.username}</span>
                        <span className="top-score">{r.score}</span>
                      </li>
                    )
                  })}
                </ol>
              </div>
            )}

            <ul className="result-buildings">
              {result.buildings.map((name) => (
                <li key={name}>{name}</li>
              ))}
            </ul>

            <button
              type="button"
              className="primary"
              onClick={handleResultDone}
            >
              Done
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
