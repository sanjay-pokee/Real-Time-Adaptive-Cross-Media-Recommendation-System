import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: false,   // fall back to next free port automatically
    open: true,
    // Let the Cloudflare quick tunnel (https://*.trycloudflare.com) reach the
    // dev/preview server; preview inherits allowedHosts and proxy from here.
    allowedHosts: ['.trycloudflare.com'],
    proxy: {
      // Proxy /api/* → http://127.0.0.1:8000/* so CORS is never an issue
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  preview: {
    port: 4173,
    open: false,
  },
})
