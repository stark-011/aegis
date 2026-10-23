const typeStyle = {
  benign: 'text-emerald-300',
  ddos: 'text-orange-300',
  sqli: 'text-rose-300',
  xss: 'text-red-300',
  portscan: 'text-yellow-300',
  malware: 'text-red-400',
  zeroday: 'text-fuchsia-300',
  grey_zone: 'text-sky-300',
}

function time(iso) { return iso ? new Date(iso).toLocaleTimeString() : '—' }

export default function Dashboard({ events, stats, error }) {
  return <section className="space-y-6">
    <div className="grid gap-4 sm:grid-cols-2">
      <StatCard label="Threats Mitigated" value={stats.mitigated} detail="Current 20-event window" alert />
      <StatCard label="Average Decision Latency" value={`${stats.averageLatency.toFixed(2)} ms`} detail="Router response time" />
    </div>
    <section className="overflow-hidden rounded-xl border border-slate-800 bg-slate-950/75 shadow-neon">
      <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
        <div><h2 className="font-semibold text-slate-100">Security Event Feed</h2><p className="mt-1 text-xs text-slate-500">Latest AEGIS traffic decisions</p></div>
        <span className="font-mono text-xs text-slate-500">{events.length}/20 EVENTS</span>
      </div>
      {error && <p className="border-b border-red-900/50 bg-red-950/30 px-5 py-3 text-sm text-red-300">API offline: {error}</p>}
      <div className="overflow-x-auto"><table className="w-full min-w-[650px] text-left text-sm">
        <thead className="bg-slate-900/70 font-mono text-[11px] uppercase tracking-wider text-slate-500"><tr><th className="px-5 py-3">Time</th><th>Source IP</th><th>Threat</th><th>Action</th><th className="pr-5 text-right">Latency</th></tr></thead>
        <tbody className="divide-y divide-slate-800/80">
          {events.length === 0 ? <tr><td colSpan="5" className="px-5 py-12 text-center font-mono text-sm text-slate-600">AWAITING TRAFFIC EVENTS...</td></tr> : events.map((event, index) => <tr key={`${event.timestamp}-${index}`} className="transition hover:bg-slate-900/60">
            <td className="px-5 py-3 font-mono text-xs text-slate-400">{time(event.timestamp)}</td><td className="font-mono text-xs text-slate-300">{event.source_ip}</td>
            <td className={`font-medium uppercase ${typeStyle[event.threat_type] || 'text-slate-300'}`}>{event.threat_type}</td>
            <td><span className={`rounded border px-2 py-1 font-mono text-[11px] ${event.action_taken === 'allow' ? 'border-emerald-800 bg-emerald-950/40 text-emerald-300' : 'border-red-900 bg-red-950/40 text-red-300'}`}>{event.action_taken}</span></td>
            <td className="pr-5 text-right font-mono text-xs text-slate-400">{Number(event.latency_ms).toFixed(3)} ms</td>
          </tr>)}</tbody>
      </table></div>
    </section>
  </section>
}

function StatCard({ label, value, detail, alert }) { return <article className="rounded-xl border border-slate-800 bg-slate-950/75 p-5 shadow-neon"><p className="font-mono text-xs uppercase tracking-wider text-slate-500">{label}</p><p className={`mt-3 text-3xl font-bold ${alert ? 'text-red-400' : 'text-emerald-400'}`}>{value}</p><p className="mt-2 text-xs text-slate-500">{detail}</p></article> }
