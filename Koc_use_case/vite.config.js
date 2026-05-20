import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import leaderboardPlugin from './vite-plugin-leaderboard.js'

export default defineConfig({
  plugins: [react(), leaderboardPlugin()],
  server: {
    port: 3000,
    open: true
  }
})
