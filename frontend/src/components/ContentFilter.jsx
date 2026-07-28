import { motion } from 'framer-motion';
import { BookOpen, Film, LayoutGrid, Music } from 'lucide-react';

const FILTERS = [
  { label: 'All', value: null, icon: LayoutGrid },
  { label: 'Movies', value: 'movie', icon: Film },
  { label: 'Books', value: 'book', icon: BookOpen },
  { label: 'Music', value: 'music', icon: Music },
];

export default function ContentFilter({ value, onChange }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {FILTERS.map(({ label, value: filterValue, icon: Icon }) => {
        const active = filterValue === value;
        return (
          <motion.button
            key={String(filterValue)}
            type="button"
            onClick={() => onChange(filterValue)}
            whileTap={{ scale: 0.98 }}
            className={`flex items-center justify-center gap-2 rounded-xl border px-3 py-2.5 text-sm font-extrabold transition ${active ? 'border-blue-600 bg-blue-600 text-white shadow-sm' : 'border-slate-200 bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50'}`}
          >
            <Icon size={15} />
            {label}
          </motion.button>
        );
      })}
    </div>
  );
}