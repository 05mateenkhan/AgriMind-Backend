const InputField = ({ label, suffix, ...props }) => (
  <label className="flex flex-col gap-2 text-sm">
    <span className="text-text-secondary">{label}</span>
    <div className="relative">
      <input
        className="w-full rounded-2xl bg-cream-light border border-border-cream focus:border-green-primary focus:ring-2 focus:ring-green-light/60 px-4 py-3 text-text-dark placeholder-text-secondary/80 transition"
        {...props}
      />
      {suffix && (
        <span className="absolute right-4 top-1/2 -translate-y-1/2 text-subtle-text text-xs">
          {suffix}
        </span>
      )}
    </div>
  </label>
);

export default InputField;

