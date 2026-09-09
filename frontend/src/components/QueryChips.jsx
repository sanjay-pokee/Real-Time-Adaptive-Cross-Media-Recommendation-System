import { motion } from 'framer-motion';

const CHIPS = [
  { label: 'Space adventure with aliens', tag: 'Sci-fi' },
  { label: 'Dark thriller with mystery', tag: 'Thriller' },
  { label: 'Romantic drama with music', tag: 'Romance' },
  { label: 'Fantasy magic adventure', tag: 'Fantasy' },
  { label: 'Motivational business books', tag: 'Learning' },
  { label: 'Energetic pop dance music', tag: 'Music' },
  { label: 'Family animation comedy', tag: 'Family' },
  { label: 'Mind-bending sci-fi thriller', tag: 'Smart' },
];

export default function QueryChips({ onSelect }) {
  return (
    <div className="flex flex-col gap-1">
      {CHIPS.map((chip, index) => (
        <motion.button
          key={chip.label}
          type="button"
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.03, duration: 0.3 }}
          onClick={() => onSelect(chip.label)}
          className="group flex items-center gap-2.5 rounded-lg px-2 py-1.5 text-left transition-colors hover:bg-surface-3"
        >
          <span className="w-16 shrink-0 text-[10px] font-bold uppercase tracking-[0.06em] text-ink-faint transition-colors group-hover:text-accent">
            {chip.tag}
          </span>
          <span className="min-w-0 flex-1 truncate text-xs text-ink-muted transition-colors group-hover:text-ink">
            {chip.label}
          </span>
        </motion.button>
      ))}
    </div>
  );
}
