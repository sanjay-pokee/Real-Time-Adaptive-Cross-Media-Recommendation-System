import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

const USERS = [
  { id: 'user_scifi', label: 'Sci-Fi Explorer', initials: 'SF', accent: '#2563EB' },
  { id: 'user_fantasy', label: 'Fantasy Wanderer', initials: 'FW', accent: '#7C3AED' },
  { id: 'user_romance', label: 'Romance Reader', initials: 'RR', accent: '#DB2777' },
  { id: 'user_action', label: 'Action Fan', initials: 'AF', accent: '#DC2626' },
  { id: 'user_music_pop', label: 'Pop Music Lover', initials: 'PM', accent: '#0891B2' },
  { id: 'user_music_rock', label: 'Rock Enthusiast', initials: 'RE', accent: '#475569' },
  { id: 'user_books_learning', label: 'Knowledge Seeker', initials: 'KS', accent: '#059669' },
  { id: 'user_family', label: 'Family & Animation', initials: 'FA', accent: '#EA580C' },
  { id: 'user_dark_thriller', label: 'Dark Thriller Addict', initials: 'DT', accent: '#111827' },
  { id: 'user_balanced', label: 'Balanced Taste', initials: 'BT', accent: '#4F46E5' },
];

export default function UserSelector({ value, onChange, users = USERS }) {
  const [open, setOpen] = useState(false);
  const [rect, setRect] = useState(null);
  const btnRef = useRef(null);
  const dropdownRef = useRef(null);

  const selected = useMemo(() => users.find(user => user.id === value) || users[0], [users, value]);

  const openDropdown = useCallback(() => {
    if (btnRef.current) setRect(btnRef.current.getBoundingClientRect());
    setOpen(true);
  }, []);

  useEffect(() => {
    if (!open) return undefined;

    function handleOutside(event) {
      const target = event.target;
      if (btnRef.current?.contains(target)) return;
      if (dropdownRef.current?.contains(target)) return;
      setOpen(false);
    }

    function handleScroll(event) {
      if (dropdownRef.current?.contains(event.target)) return;
      setOpen(false);
    }

    document.addEventListener('pointerdown', handleOutside);
    window.addEventListener('scroll', handleScroll);

    return () => {
      document.removeEventListener('pointerdown', handleOutside);
      window.removeEventListener('scroll', handleScroll);
    };
  }, [open]);

  const dropdown = open && rect && createPortal(
    <AnimatePresence>
      <motion.div
        ref={dropdownRef}
        key="user-dropdown"
        initial={{ opacity: 0, y: -6, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -6, scale: 0.98 }}
        transition={{ type: 'spring', stiffness: 420, damping: 30 }}
        style={{
          position: 'fixed',
          top: rect.bottom + 8,
          left: rect.left,
          width: Math.max(rect.width, 260),
          zIndex: 99999,
        }}
        className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-soft"
      >
        <div className="max-h-[280px] overflow-y-auto p-1.5" style={{ touchAction: 'pan-y' }}>
          {users.map(user => (
            <button
              key={user.id}
              type="button"
              onPointerDown={event => {
                event.stopPropagation();
                onChange(user.id);
                setOpen(false);
              }}
              className={`flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition ${user.id === value ? 'bg-blue-50 text-blue-900' : 'text-slate-700 hover:bg-slate-50'}`}
            >
              <span className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl text-xs font-black text-white" style={{ background: user.accent }}>
                {user.initials}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-extrabold">{user.label}</span>
                <span className="block truncate text-xs font-semibold text-slate-400">{user.id}</span>
              </span>
              {user.id === value && <span className="h-2 w-2 rounded-full bg-blue-600" />}
            </button>
          ))}
        </div>
      </motion.div>
    </AnimatePresence>,
    document.body
  );

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        onPointerDown={event => {
          event.stopPropagation();
          if (open) setOpen(false);
          else openDropdown();
        }}
        className="app-input flex items-center gap-3 px-3 py-2.5 text-sm font-semibold hover:border-slate-300"
      >
        <span className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg text-xs font-black text-white" style={{ background: selected.accent }}>
          {selected.initials}
        </span>
        <span className="min-w-0 flex-1 truncate text-left text-slate-900">{selected.label}</span>
        <ChevronDown size={15} className="flex-shrink-0 text-slate-400" />
      </button>
      {dropdown}
    </>
  );
}

export { USERS };