import { AnimatePresence, motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Info, X, XCircle } from 'lucide-react';
import { useEffect } from 'react';

const TONES = {
  success: { Icon: CheckCircle, color: 'var(--ok)' },
  error: { Icon: XCircle, color: 'var(--bad)' },
  info: { Icon: Info, color: 'var(--accent)' },
  warning: { Icon: AlertTriangle, color: 'var(--warn)' },
};

export default function Toast({ toasts = [], onDismiss }) {
  return (
    <div
      className="pointer-events-none fixed bottom-5 right-5 z-[9999] flex w-[calc(100%-2.5rem)] max-w-xs flex-col gap-2"
      role="status"
      aria-live="polite"
    >
      <AnimatePresence>
        {toasts.map((toast) => (
          <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
        ))}
      </AnimatePresence>
    </div>
  );
}

function ToastItem({ toast, onDismiss }) {
  useEffect(() => {
    const timer = setTimeout(() => onDismiss(toast.id), toast.duration || 4000);
    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, onDismiss]);

  const { Icon, color } = TONES[toast.type] || TONES.info;

  return (
    <motion.div
      initial={{ x: 60, opacity: 0, scale: 0.97 }}
      animate={{ x: 0, opacity: 1, scale: 1 }}
      exit={{ x: 60, opacity: 0, scale: 0.96 }}
      transition={{ type: 'spring', stiffness: 420, damping: 32 }}
      className="panel pointer-events-auto flex items-start gap-2.5 overflow-hidden p-3 shadow-lg"
    >
      <span
        className="absolute inset-y-0 left-0 w-[3px]"
        style={{ background: color }}
        aria-hidden="true"
      />
      <Icon size={14} className="mt-px shrink-0" style={{ color }} />
      <div className="min-w-0 flex-1">
        {toast.title && <p className="text-[12px] font-semibold text-ink">{toast.title}</p>}
        <p className="text-[11.5px] leading-relaxed text-ink-muted">{toast.message}</p>
      </div>
      <button
        type="button"
        onClick={() => onDismiss(toast.id)}
        aria-label="Dismiss"
        className="shrink-0 text-ink-faint transition-colors hover:text-ink"
      >
        <X size={12} />
      </button>
    </motion.div>
  );
}
