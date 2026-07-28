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
    <div className="flex flex-wrap gap-2">
      {CHIPS.map((chip, index) => (
        <motion.button
          key={chip.label}
          type="button"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.035 }}
          whileHover={{ y: -1 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onSelect(chip.label)}
          className="rounded-full border border-slate-200 bg-white px-3 py-2 text-left text-xs font-extrabold text-slate-600 shadow-sm transition hover:border-blue-200 hover:bg-blue-50 hover:text-blue-700"
        >
          <span className="mr-2 text-slate-400">{chip.tag}</span>
          {chip.label}
        </motion.button>
      ))}
    </div>
  );
}