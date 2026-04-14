import { motion } from 'framer-motion';

const LoadingOverlay = ({ label = 'Processing request...' }) => (
  <div className="absolute inset-0 flex items-center justify-center bg-cream-light/80 backdrop-blur-lg rounded-3xl">
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex flex-col items-center gap-4"
    >
      <motion.div
        className="w-16 h-16 rounded-full border-4 border-border-cream border-t-green-accent"
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}
      />
      <p className="text-brand-muted-text font-medium">{label}</p>
    </motion.div>
  </div>
);

export default LoadingOverlay;

