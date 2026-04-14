import { motion } from 'framer-motion';

const variants = {
  hidden: { opacity: 0, y: 40 },
  visible: { opacity: 1, y: 0 },
};

const AnimatedSection = ({ children, delay = 0 }) => (
  <motion.section
    initial="hidden"
    animate="visible"
    exit="hidden"
    variants={variants}
    transition={{ duration: 0.7, delay }}
    className="w-full"
  >
    {children}
  </motion.section>
);

export default AnimatedSection;

