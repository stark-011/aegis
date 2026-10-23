import { useCallback, useEffect, useMemo, useState } from 'react'
import Dashboard from './components/Dashboard'
import Simulator from './components/Simulator'
import { api } from './api'

export default function App() {
  const [events, setEvents] = useState([])
  const [consoleEntry, setConsoleEntry] = useState(null)
  const [apiError, setApiError] = useState('')

  const refreshLogs = useCallback(async () => {
    try {
      setEvents(await api.getLogs())
      setApiError('')
    } catch (error) { setApiError(error.message) }
  }, [])

  useEffect(() => {
    refreshLogs()
    const interval = setInterval(refreshLogs, 5000)
    return () => clearInterval(interval)
  }, [refreshLogs])

  const stats = useMemo(() => {
    const mitigated = events.filter((event) => event.action_taken !== 'allow')
    const averageLatency = events.length
      ? events.reduce((sum, event) => sum + Number(event.latency_ms || 0), 0) / events.length : 0
    return { mitigated: mitigated.length, averageLatency }
  }, [events])

  async function simulate(threatType) {
    try {
      const response = await api.simulate(threatType)
      setConsoleEntry(response)
      setEvents((current) => [response, ...current].slice(0, 20))
      setApiError('')
      await refreshLogs()
    } catch (error) {
      setConsoleEntry({ status: 'REQUEST FAILED', error: error.message })
      setApiError(error.message)
    }
  }

  return (
    <main className="scanline min-h-screen px-4 py-6 sm:px-8 lg:px-12">
      <header className="mx-auto mb-7 flex max-w-7xl flex-col gap-3 border-b border-slate-800 pb-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="mb-1 font-mono text-xs tracking-[.28em] text-emerald-400">AI AUTO DEFENSE PLATFORM</p>
          <h1 className="text-3xl font-bold tracking-tight text-slate-100">AEGIS SOC Command</h1>
        </div>
        <div className="flex items-center gap-2 font-mono text-xs text-emerald-300"><span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" /> DEFENSE GRID ONLINE</div>
      </header>
      <div className="mx-auto grid max-w-7xl gap-6 xl:grid-cols-[1.45fr_.9fr]">
        <Dashboard events={events} stats={stats} error={apiError} />
        <Simulator onSimulate={simulate} consoleEntry={consoleEntry} />
      </div>
    </main>
  )
}
