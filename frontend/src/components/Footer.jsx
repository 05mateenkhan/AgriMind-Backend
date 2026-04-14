const Footer = () => {
  const year = new Date().getFullYear();

  return (
    <footer className="px-4 pb-10">
      <div className="max-w-6xl mx-auto text-sm text-brand-subtle-text text-center">
        <p>© {year} AgriMind Intelligence. Designed for resilient farming.</p>
      </div>
    </footer>
  );
};

export default Footer;

