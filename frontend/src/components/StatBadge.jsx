const StatBadge = ({ value, label }) => (
  <div className="flex flex-col gap-1 px-4 py-3 rounded-2xl bg-cream-card border border-border-cream shadow-sm">
    <span className="text-2xl font-semibold text-green-primary">{value}</span>
    <span className="text-xs uppercase tracking-wide text-text-secondary">
      {label}
    </span>
  </div>
);

export default StatBadge;

