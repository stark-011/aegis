import { useState } from 'react'

const scenarios = [
  { type: 'benign', label: 'Simulate Benign Traffic', style: 'border-emerald-800 bg-emerald-950/30 text-emerald-200 hover:bg-emerald-900/45' },
  { type: 'ddos', label: 'Trigger DDoS SYN Flood', style: 'border-orange-800 bg-orange-950/30 text-orange-200 hover:bg-orange-900/45' },
  { type: 'sqli', label: 'Launch SQL Injection', style: 'border-rose-800 bg-rose-950/30 text-rose-200 hover:bg-rose-900/45' },
  { type: 'xss', label: 'Execute Cross-Site Scripting (XSS)', style: 'border-red-800 bg-red-950/30 text-red-200 hover:bg-red-900/45' },
  { type: 'portscan', label: 'Launch Port Scan Reconnaissance', style: 'border-yellow-700 bg-yellow-950/30 text-yellow-200 hover:bg-yellow-900/45' },
  { type: 'malware', label: 'Simulate Mirai IoT Malware', style: 'border-red-950 bg-red-950/50 text-red-100 hover:bg-red-900/60' },
  { type: 'zeroday', label: 'Inject Zero-Day Anomaly', style: 'border-fuchsia-800 bg-fuchsia-950/30 text-fuchsia-200 hover:bg-fuchsia-900/45' },
  { type: 'grey_zone', label: 'Trigger Grey-Zone Ambiguity', style: 'border-sky-800 bg-sky-950/30 text-sky-200 hover:bg-sky-900/45' },
]

export default function Simulator({ onSimulate, consoleEntry }) {
  const [running, setRunning] = useState('')
  async function run(type) { setRunning(type); await onSimulate(type); setRunning('') }
  const terminal = consoleEntry ? JSON.stringify(consoleEntry, null, 2) : 'AEGIS simulator ready. Select a traffic profile to begin.'
  return <aside className="space-y-6">
    <section className="rounded-xl border border-slate-800 bg-slate-950/75 p-5 shadow-neon"><div className="mb-4"><h2 className="font-semibold text-slate-100">Attack Simulator</h2><p className="mt-1 text-xs leading-5 text-slate-500">Synthetic flows are evaluated by the live AEGIS response policy.</p></div><div className="grid gap-3">{scenarios.map((scenario) => <button key={scenario.type} disabled={Boolean(running)} onClick={() => run(scenario.type)} className={`rounded-lg border px-4 py-3 text-left text-sm font-medium transition disabled:cursor-wait disabled:opacity-50 ${scenario.style}`}><span className="mr-2 font-mono text-xs opacity-60">&gt;_</span>{running === scenario.type ? 'Analyzing traffic...' : scenario.label}</button>)}</div></section>
    <section className="overflow-hidden rounded-xl border border-slate-800 bg-[#030807] shadow-neon"><div className="flex items-center gap-2 border-b border-slate-800 bg-slate-950 px-4 py-3"><span className="h-2.5 w-2.5 rounded-full bg-red-500"/><span className="h-2.5 w-2.5 rounded-full bg-amber-400"/><span className="h-2.5 w-2.5 rounded-full bg-emerald-400"/><span className="ml-2 font-mono text-[11px] text-slate-500">AEGIS ACTION CONSOLE</span></div><pre className="min-h-72 overflow-auto whitespace-pre-wrap break-words p-4 font-mono text-xs leading-6 text-emerald-300">{terminal}</pre></section>
  </aside>
}
