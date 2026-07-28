import { motion } from 'framer-motion';

export default function GlassPanel({ children, className = '', variant = 'default', ...props }) {
  const base =
    variant === 'strong'
      ? 'glass-strong'
      : variant === 'card'
      ? 'glass-card'
      : 'glass';

  return (
    <motion.div
      className={`${base} ${className}`}
      {...props}
    >
      {children}
    </motion.div>
  );
}
