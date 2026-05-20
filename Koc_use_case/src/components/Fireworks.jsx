import { useEffect, useRef } from 'react'
import './Fireworks.css'

const COLORS = [
  '#ff3b6b',
  '#ffb44a',
  '#ffe14a',
  '#3bd56b',
  '#3bb0ff',
  '#9d6bff',
  '#ff6bdc',
  '#ffffff'
]

function pickColor() {
  return COLORS[Math.floor(Math.random() * COLORS.length)]
}

export default function Fireworks() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    let raf = 0
    let running = true
    const dpr = window.devicePixelRatio || 1

    const resize = () => {
      canvas.width = window.innerWidth * dpr
      canvas.height = window.innerHeight * dpr
      canvas.style.width = window.innerWidth + 'px'
      canvas.style.height = window.innerHeight + 'px'
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    window.addEventListener('resize', resize)

    const particles = []

    const createBurst = (x, y) => {
      const count = 40 + Math.floor(Math.random() * 30)
      const color = pickColor()
      const speed = 4.5 + Math.random() * 2.5
      for (let i = 0; i < count; i++) {
        const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.3
        const v = speed * (0.5 + Math.random() * 0.7)
        particles.push({
          x,
          y,
          vx: Math.cos(angle) * v,
          vy: Math.sin(angle) * v,
          life: 1,
          decay: 0.012 + Math.random() * 0.012,
          color,
          size: 1.8 + Math.random() * 2.2
        })
      }
    }

    let lastBurst = 0
    const tick = (t) => {
      if (!running) return

      ctx.fillStyle = 'rgba(0, 0, 0, 0.18)'
      ctx.fillRect(0, 0, window.innerWidth, window.innerHeight)

      if (t - lastBurst > 350 + Math.random() * 450) {
        const x = window.innerWidth * (0.12 + Math.random() * 0.76)
        const y = window.innerHeight * (0.1 + Math.random() * 0.55)
        createBurst(x, y)
        lastBurst = t
      }

      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i]
        p.x += p.vx
        p.y += p.vy
        p.vy += 0.06
        p.vx *= 0.99
        p.life -= p.decay
        if (p.life <= 0) {
          particles.splice(i, 1)
          continue
        }
        ctx.globalAlpha = p.life
        ctx.fillStyle = p.color
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
        ctx.fill()
      }
      ctx.globalAlpha = 1

      raf = requestAnimationFrame(tick)
    }

    let delay = 0
    for (let i = 0; i < 4; i++) {
      const t = setTimeout(() => {
        createBurst(
          window.innerWidth * (0.2 + Math.random() * 0.6),
          window.innerHeight * (0.18 + Math.random() * 0.4)
        )
      }, delay)
      delay += 180
    }

    raf = requestAnimationFrame(tick)

    return () => {
      running = false
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return <canvas ref={canvasRef} className="fireworks-canvas" aria-hidden="true" />
}
