import { motion } from 'framer-motion';

/**
 * Standard elevated surface. `variant="strong"` adds the lit top edge used on
 * hero-level panels; `variant="card"` opts into the hover-lift treatment.
 */
export default function GlassPanel({ children, className = '', variant = 'default', ...props }) {
  const base =
    variant === 'strong' ? 'panel panel-lit' : variant === 'card' ? 'card' : 'panel';

  return (
    <motion.div className={`${base} ${className}`} {...props}>
      {children}
    </motion.div>
  );
}
