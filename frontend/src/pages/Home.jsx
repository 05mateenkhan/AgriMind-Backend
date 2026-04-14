import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import GlassCard from '../components/GlassCard.jsx';
import StatBadge from '../components/StatBadge.jsx';
import AnimatedSection from '../components/AnimatedSection.jsx';

const Home = () => {
  return (
    <div className="max-w-6xl mx-auto space-y-12">
      <AnimatedSection>
        <section className="grid lg:grid-cols-2 gap-10 items-center">
          <div className="space-y-8">
            <motion.p
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cream-highlight text-text-olive text-sm font-medium"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              Intelligent crop & disease insights
            </motion.p>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="space-y-6"
            >
              <h1 className="font-display text-4xl md:text-5xl leading-tight">
                Reimagine farming with{' '}
                <span className="text-green-primary">AI</span>
              </h1>
              <p className="text-lg text-text-secondary">
                AgriMind blends agronomic intelligence, climate data, and edge
                vision to deliver precise crop recommendations and real-time
                disease alerts.
              </p>
            </motion.div>
            <div className="flex flex-wrap gap-4">
              <Link
                to="/smart-crop"
                className="px-6 py-3 rounded-full bg-green-button text-text-olive font-semibold shadow-glow hover:bg-green-secondary hover:-translate-y-0.5 transition"
              >
                Smart Crop Recommendation
              </Link>
              <Link
                to="/disease-detection"
                className="px-6 py-3 rounded-full border border-border-cream text-text-dark hover:bg-cream-card transition"
              >
                Crop Disease Detection
              </Link>
            </div>
          </div>

          <GlassCard className="relative overflow-hidden shadow-soft">
            <div className="absolute inset-0 bg-hero-gradient opacity-90" />
            <div className="relative space-y-6">
              <h3 className="text-2xl font-semibold">
                Climate-aware intelligence
              </h3>
              <p className="text-text-secondary">
                Seamlessly monitor soil nutrients, micro-climate signals, and
                visual plant health to make reliable decisions throughout the
                season.
              </p>
              <div className="grid grid-cols-2 gap-4">
                <StatBadge value="120+" label="Crop insights" />
                <StatBadge value="95%" label="Model accuracy" />
                <StatBadge value="24hr" label="Fresh data sync" />
                <StatBadge value="30+" label="Disease classes" />
              </div>
            </div>
          </GlassCard>
        </section>
      </AnimatedSection>

      <AnimatedSection delay={0.2}>
        <section className="grid md:grid-cols-2 gap-6">
          <GlassCard className="space-y-4 shadow-soft">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full bg-green-light flex items-center justify-center text-text-olive">
                <span className="text-yellow-wheat">🌿</span>
              </div>
              <div>
                <p className="text-sm text-text-secondary/80">Module 01</p>
                <h4 className="text-xl font-semibold">Smart Crop Lab</h4>
              </div>
            </div>
            <p className="text-text-secondary">
              Blend soil chemistry and weather parameters to discover the most
              profitable crop for your parcel before sowing.
            </p>
            <Link
              to="/smart-crop"
              className="inline-flex items-center gap-2 text-green-primary font-semibold"
            >
              Explore tool →
            </Link>
          </GlassCard>
          <GlassCard className="space-y-4 shadow-soft">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-full bg-green-light flex items-center justify-center text-text-olive">
                <span className="text-yellow-corn">🔬</span>
              </div>
              <div>
                <p className="text-sm text-text-secondary/80">Module 02</p>
                <h4 className="text-xl font-semibold">Disease Vision AI</h4>
              </div>
            </div>
            <p className="text-text-secondary">
              Upload a field photo and receive early disease signatures,
              confidence, and actionable agronomy guidance.
            </p>
            <Link
              to="/disease-detection"
              className="inline-flex items-center gap-2 text-green-primary font-semibold"
            >
              Scan now →
            </Link>
          </GlassCard>
        </section>
      </AnimatedSection>
    </div>
  );
};

export default Home;

