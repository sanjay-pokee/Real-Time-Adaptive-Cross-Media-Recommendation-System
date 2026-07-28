import { motion } from 'framer-motion';
import { Loader2, RefreshCw, WifiOff } from 'lucide-react';

export default function BackendStatus({ status, onRetry }) {
  return (
    <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} className="flex items-center gap-2">
      {status === 'checking' && (
        <div className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5">
          <Loader2 size={13} className="animate-spin text-blue-600" />
          <span className="text-xs font-extrabold text-slate-500">Checking</span>
        </div>
      )}

      {status === 'online' && (
        <div className="flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-emerald-500" />
          <span className="text-xs font-extrabold text-emerald-700">Online</span>
        </div>
      )}

      {status === 'offline' && (
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 rounded-full border border-red-200 bg-red-50 px-3 py-1.5">
            <WifiOff size={13} className="text-red-600" />
            <span className="text-xs font-extrabold text-red-700">Offline</span>
          </div>
          {onRetry && (
            <button type="button" onClick={onRetry} className="flex items-center gap-1 rounded-full border border-slate-200 bg-white px-2.5 py-1.5 text-xs font-extrabold text-slate-500 transition hover:text-blue-700">
              <RefreshCw size={11} />
              Retry
            </button>
          )}
        </div>
      )}
    </motion.div>
  );
}