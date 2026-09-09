import { motion } from 'framer-motion';
import { Loader2, RefreshCw, WifiOff } from 'lucide-react';

export default function BackendStatus({ status, onRetry }) {
  if (status === 'checking') {
    return (
      <span className="chip">
        <Loader2 size={11} className="animate-spin text-accent" />
        Connecting
      </span>
    );
  }

  if (status === 'online') {
    return (
      <span className="chip" style={{ color: 'var(--ok)', borderColor: 'color-mix(in srgb, var(--ok) 34%, transparent)' }}>
        <span className="relative flex h-1.5 w-1.5">
          <motion.span
            className="absolute inline-flex h-full w-full rounded-full"
            style={{ background: 'var(--ok)' }}
            animate={{ opacity: [0.9, 0.25, 0.9], scale: [1, 1.9, 1] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full" style={{ background: 'var(--ok)' }} />
        </span>
        Live
      </span>
    );
  }

  return (
    <div className="flex items-center gap-1.5">
      <span className="chip" style={{ color: 'var(--bad)', borderColor: 'color-mix(in srgb, var(--bad) 34%, transparent)' }}>
        <WifiOff size={11} />
        Offline
      </span>
      {onRetry && (
        <button type="button" onClick={onRetry} className="chip chip-interactive" title="Retry connection">
          <RefreshCw size={10} />
        </button>
      )}
    </div>
  );
}
