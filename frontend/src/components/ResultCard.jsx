import { motion } from 'framer-motion';
import GlassCard from './GlassCard.jsx';

const ResultCard = ({ title, badge, description, extra }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5 }}
  >
    <GlassCard className="space-y-4 border border-border-cream shadow-soft">
      <div className="flex items-center gap-3">
        <div className="w-12 h-12 rounded-2xl bg-yellow-corn/30 flex items-center justify-center text-text-olive font-semibold">
          🌱
        </div>
        <div>
          <p className="text-sm uppercase tracking-wide text-green-primary">
            Recommendation
          </p>
          <h3 className="text-2xl font-semibold text-black">{title.toUpperCase()}</h3>
        </div>
      </div>
      {badge && (
        <span className="inline-flex px-3 py-1 rounded-full text-xs bg-yellow-corn/40 text-text-olive">
          {badge}
        </span>
      )}
      {description && (
        <p className="text-text-secondary leading-relaxed">{description}</p>
      )}
      {extra}
    </GlassCard>
  </motion.div>
);

export default ResultCard;

