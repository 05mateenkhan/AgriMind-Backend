const GlassCard = ({ children, className = '' }) => (
  <div
    className={`glass rounded-3xl p-6 gradient-border bg-card-glass ${className}`}
  >
    {children}
  </div>
);

export default GlassCard;

