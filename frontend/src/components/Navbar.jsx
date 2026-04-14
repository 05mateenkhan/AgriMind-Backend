import { useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';

const navLinks = [
  { path: '/', label: 'Home' },
  { path: '/smart-crop', label: 'Smart Crop' },
  { path: '/disease-detection', label: 'Disease Detection' },
];

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  const renderLinks = (onClick) =>
    navLinks.map((link) => (
      <NavLink
        key={link.path}
        to={link.path}
        onClick={onClick}
        className={({ isActive }) =>
          `px-4 py-2 rounded-full text-sm font-medium transition-colors ${
            isActive
              ? 'bg-green-light text-text-olive'
              : 'text-text-secondary hover:text-dark'
          }`
        }
      >
        {link.label}
      </NavLink>
    ));

  return (
    <header className="fixed top-4 left-0 w-full z-50 px-4">
      <div className="max-w-6xl mx-auto glass rounded-2xl gradient-border shadow-soft">
        <div className="flex items-center justify-between px-6 py-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-full bg-green-light flex items-center justify-center text-text-olive font-semibold">
              FV
            </div>
            <div>
              <p className="font-display text-lg leading-tight">FARM VISION</p>
              <p className="text-xs text-text-secondary">
                Future of Farming
              </p>
            </div>
          </Link>

          <nav className="hidden md:flex items-center gap-2">
            {renderLinks()}
          </nav>

          <button
            onClick={() => setIsOpen((prev) => !prev)}
            className="md:hidden p-2 rounded-full bg-cream-card hover:bg-cream-light transition shadow-sm"
            aria-label="Toggle navigation"
          >
            <div className="w-6 h-0.5 bg-brand-charcoal mb-1" />
            <div className="w-6 h-0.5 bg-brand-charcoal" />
          </button>
        </div>

        <AnimatePresence>
          {isOpen && (
            <motion.nav
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="md:hidden flex flex-col px-6 pb-4"
            >
              {renderLinks(() => setIsOpen(false))}
            </motion.nav>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
};

export default Navbar;

