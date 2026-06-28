import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Dev: serve live fires.json from the Python server (:5001) instead of the
    // static public/data seed, so the dashboard and "Update Incidents" use fresh data.
    proxy: {
      '/data': 'http://localhost:5001',
    },
  },
})