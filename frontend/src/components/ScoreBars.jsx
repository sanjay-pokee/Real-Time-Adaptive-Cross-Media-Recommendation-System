import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';

function ScoreBar({ label, value, color, delay = 0, missingText = 'Not available' }) {
  const [width, setWidth] = useState(0);
  const numericValue = value == null || value === '' ? null : Number(value);
  const hasValue = numericValue != null && Number.isFinite(numericValue);
  const pct = hasValue ? Math.max(0, Math.min(100, Math.round(numericValue * 100))) : null;

  useEffect(() => {
    const timer = setTimeout(() => setWidth(pct ?? 0), 80 + delay);
    return () => clearTimeout(timer);
  }, [pct, delay]);

  if (!hasValue) {
    return (
      <div className="flex items-center gap-3">
        <span className="w-24 flex-shrink-0 text-xs font-extrabold text-slate-500">{label}</span>
        <span className="text-xs font-semibold italic text-slate-400">{missingText}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <span className="w-24 flex-shrink-0 text-xs font-extrabold text-slate-500">{label}</span>
      <div className="score-track flex-1">
        <motion.div
          className="h-full rounded-full"
          style={{ background: color }}
          initial={{ width: 0 }}
          animate={{ width: `${width}%` }}
          transition={{ duration: 0.7, delay: delay / 1000, ease: [0.4, 0, 0.2, 1] }}
        />
      </div>
      <span className="w-10 text-right text-xs font-black" style={{ color }}>
        {numericValue.toFixed(2)}
      </span>
    </div>
  );
}

export default function ScoreBars({ score, semantic_score, graph_score, ema_score }) {
  return (
    <div className="mt-4 flex flex-col gap-2.5 border-t border-slate-100 pt-4">
      <ScoreBar label="Final" value={score} color="#2563EB" delay={0} missingText="No final score" />
      <ScoreBar label="Semantic" value={semantic_score} color="#7C3AED" delay={80} missingText="Not returned" />
      <ScoreBar label="Graph" value={graph_score} color="#059669" delay={160} missingText="No graph signal" />
      <ScoreBar label="EMA" value={ema_score} color="#D97706" delay={240} missingText="No EMA signal" />
    </div>
  );
}