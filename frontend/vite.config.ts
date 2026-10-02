import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const API_TARGET = process.env.REPROSCOPE_API ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // SSE needs an unbuffered proxy; Vite's http-proxy streams by default.
      '/api': { target: API_TARGET, changeOrigin: true },
    },
  },
})
