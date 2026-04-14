import { Route, Routes } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import Navbar from './components/Navbar.jsx';
import Home from './pages/Home.jsx';
import SmartCrop from './pages/SmartCrop.jsx';
import DiseaseDetection from './pages/DiseaseDetection.jsx';
import Footer from './components/Footer.jsx';

const App = () => {
  return (
    <div className="min-h-screen bg-green-light bg-grid text-charcoal">
      <Navbar />
      <main className="pt-28 pb-20 px-4">
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/smart-crop" element={<SmartCrop />} />
            <Route path="/disease-detection" element={<DiseaseDetection />} />
          </Routes>
        </AnimatePresence>
      </main>
      <Footer />
    </div>
  );
};

export default App;

