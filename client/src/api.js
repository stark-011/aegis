const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000').replace(/\/$/, '')

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options)
  const data = await response.json().catch(() => ({ error: 'The API returned an invalid response.' }))
  if (!response.ok && !data.action_taken) throw new Error(data.error || `Request failed (${response.status})`)
  return data
}

export const api = {
  getLogs: () => request('/api/logs'),
  simulate: (threatType) => request('/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threat_type: threatType }),
  }),
}
