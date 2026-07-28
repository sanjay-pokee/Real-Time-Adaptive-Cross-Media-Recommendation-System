import { AnimatePresence, motion } from 'framer-motion';
import { useState } from 'react';
import { Activity, Brain, ChevronDown, Combine, HelpCircle, Network, Server, User } from 'lucide-react';

const SIGNALS = [
  { icon: Brain, color: '#7C3AED', label: 'Semantic Score', key: 'semantic_score', desc: 'Query-to-content match from vector search.' },
  { icon: Network, color: '#059669', label: 'Graph Signal', key: 'graph_score', desc: 'Collaborative signal from user-item interaction embeddings.' },
  { icon: Activity, color: '#D97706', label: 'EMA Signal', key: 'ema_score', desc: 'Real-time preference signal from recent interactions.' },
  { icon: Combine, color: '#2563EB', label: 'Final Score', key: 'score', desc: 'Hybrid ranking score used to order results.' },
];

export default function AISignalsPanel({ userId, backendStatus, topResult }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm">
      <button type="button" onClick={() => setOpen(value => !value)} className="flex w-full items-center gap-3 px-5 py-4 text-left transition hover:bg-slate-50">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-blue-50 text-blue-700">
          <Brain size={17} />
        </div>
        <span className="flex-1 text-sm font-black text-slate-900">AI signals</span>
        <motion.div animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown size={16} className="text-slate-400" />
        </motion.div>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="space-y-4 border-t border-slate-100 px-5 py-4">
              <div className="grid gap-2">
                <div className="flex items-center gap-2 rounded-2xl bg-slate-50 px-3 py-2 text-xs font-bold text-slate-500">
                  <User size={13} className="text-blue-600" />
                  <span className="flex-1">Active user</span>
                  <span className="font-black text-slate-900">{userId}</span>
                </div>
                <div className="flex items-center gap-2 rounded-2xl bg-slate-50 px-3 py-2 text-xs font-bold text-slate-500">
                  <Server size={13} className={backendStatus === 'online' ? 'text-emerald-600' : 'text-red-600'} />
                  <span className="flex-1">Backend</span>
                  <span className={backendStatus === 'online' ? 'font-black text-emerald-700' : 'font-black text-red-700'}>{backendStatus}</span>
                </div>
              </div>

              {SIGNALS.map(({ icon: Icon, color, label, key, desc }) => (
                <div key={key} className="flex gap-3">
                  <div className="mt-0.5 flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-xl" style={{ background: `${color}14`, color }}>
                    <Icon size={15} />
                  </div>
                  <div className="min-w-0">
                    <div className="mb-0.5 flex flex-wrap items-center gap-2">
                      <span className="text-xs font-black" style={{ color }}>{label}</span>
                      {topResult?.[key] != null && <span className="rounded-lg bg-slate-100 px-1.5 py-0.5 text-xs font-black text-slate-700">{Number(topResult[key]).toFixed(3)}</span>}
                      {topResult && topResult[key] == null && (key === 'graph_score' || key === 'ema_score') && <span className="text-[11px] font-semibold italic text-slate-400">not computed</span>}
                    </div>
                    <p className="text-xs leading-5 text-slate-500">{desc}</p>
                  </div>
                </div>
              ))}

              <div className="flex gap-2 rounded-2xl border border-blue-100 bg-blue-50 px-3 py-2.5">
                <HelpCircle size={14} className="mt-0.5 flex-shrink-0 text-blue-700" />
                <p className="text-xs font-semibold leading-5 text-blue-900">Interactions update personalization signals when the backend is running.</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}