import fs from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'

const require = createRequire(import.meta.url)
const XLSX = require('xlsx')

const FILE_NAME = 'leaderboard.xlsx'
const SHEET_NAME = 'Leaderboard'

function readRecords(filePath) {
  if (!fs.existsSync(filePath)) return []
  try {
    const wb = XLSX.readFile(filePath)
    const sheet = wb.Sheets[wb.SheetNames[0]]
    if (!sheet) return []
    return XLSX.utils.sheet_to_json(sheet, { defval: '' })
  } catch (err) {
    console.error('[leaderboard] failed to read xlsx:', err)
    return []
  }
}

function writeRecords(filePath, records) {
  const ws = XLSX.utils.json_to_sheet(records, {
    header: ['username', 'buildings', 'score', 'timestamp']
  })
  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, ws, SHEET_NAME)
  XLSX.writeFile(wb, filePath)
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = ''
    req.on('data', (chunk) => {
      data += chunk
    })
    req.on('end', () => resolve(data))
    req.on('error', reject)
  })
}

function sendJson(res, status, payload) {
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json')
  res.end(JSON.stringify(payload))
}

export default function leaderboardPlugin() {
  return {
    name: 'campus-leaderboard',
    configureServer(server) {
      const filePath = path.resolve(server.config.root, FILE_NAME)

      server.middlewares.use('/api/leaderboard', async (req, res) => {
        try {
          if (req.method === 'GET') {
            const records = readRecords(filePath)
            sendJson(res, 200, records)
            return
          }
          if (req.method === 'POST') {
            const raw = await readBody(req)
            let record
            try {
              record = JSON.parse(raw)
            } catch {
              sendJson(res, 400, { error: 'Invalid JSON' })
              return
            }
            if (
              typeof record.username !== 'string' ||
              typeof record.score !== 'number'
            ) {
              sendJson(res, 400, { error: 'username and score are required' })
              return
            }
            const records = readRecords(filePath)
            const row = {
              username: record.username,
              buildings: record.buildings ?? '',
              score: record.score,
              timestamp: record.timestamp ?? new Date().toISOString()
            }
            records.push(row)
            writeRecords(filePath, records)
            sendJson(res, 201, { ok: true, count: records.length, record: row })
            return
          }
          res.statusCode = 405
          res.setHeader('Allow', 'GET, POST')
          res.end('Method Not Allowed')
        } catch (err) {
          console.error('[leaderboard] request failed:', err)
          sendJson(res, 500, { error: err.message })
        }
      })
    }
  }
}
