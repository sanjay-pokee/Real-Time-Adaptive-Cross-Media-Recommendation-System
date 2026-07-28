import { AnimatePresence, motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, Info, X, XCircle } from 'lucide-react';
import { useEffect } from 'react';

const ICONS = {
  success: <CheckCircle size={18} className="text-emerald-600" />,
  error: <XCircle size={18} className="text-red-600" />,
  info: <Info size={18} className="text-blue-600" />,
  warning: <AlertTriangle size={18} className="text-amber-600" />,
};

const COLORS = {
  success: 'border-l-emerald-500 bg-emerald-50',
  error: 'border-l-red-500 bg-red-50',
  info: 'border-l-blue-500 bg-blue-50',
  warning: 'border-l-amber-500 bg-amber-50',
};

export default function Toast({ toasts = [], onDismiss }) {
  return (
    <div className="pointer-events-none fixed bottom-6 right-6 z-[9999] flex w-[calc(100%-2rem)] max-w-sm flex-col gap-3">
      <AnimatePresence>
        {toasts.map(toast => <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />)}
      </AnimatePresence>
    </div>
  );
}

function ToastItem({ toast, onDismiss }) {
  useEffect(() => {
    const timer = setTimeout(() => onDismiss(toast.id), toast.duration || 4000);
    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, onDismiss]);

  return (
    <motion.div
      initial={{ x: 80, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 80, opacity: 0, scale: 0.96 }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      className={`pointer-events-auto flex items-start gap-3 rounded-2xl border border-slate-200 border-l-4 px-4 py-3 shadow-soft ${COLORS[toast.type] || COLORS.info}`}
    >
      <span className="mt-0.5 flex-shrink-0">{ICONS[toast.type] || ICONS.info}</span>
      <div className="min-w-0 flex-1">
        {toast.title && <p className="text-sm font-black text-slate-950">{toast.title}</p>}
        <p className="mt-0.5 text-xs font-semibold leading-5 text-slate-600">{toast.message}</p>
      </div>
      <button type="button" onClick={() => onDismiss(toast.id)} className="mt-0.5 flex-shrink-0 text-slate-400 transition hover:text-slate-800">
        <X size={14} />
      </button>
    </motion.div>
  );
}
