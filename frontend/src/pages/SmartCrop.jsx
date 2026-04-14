import { useState } from 'react';
import { motion } from 'framer-motion';
import GlassCard from '../components/GlassCard.jsx';
import InputField from '../components/InputField.jsx';
import ResultCard from '../components/ResultCard.jsx';
import LoadingOverlay from '../components/LoadingOverlay.jsx';
import AnimatedSection from '../components/AnimatedSection.jsx';
import { predictCrop } from '../services/api.js';

const defaultForm = {
  N: '',
  P: '',
  K: '',
  temperature: '',
  humidity: '',
  ph: '',
  rainfall: '',
};

const SmartCrop = () => {
  const [formValues, setFormValues] = useState(defaultForm);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await predictCrop(formValues);
      setResult(data);
    } catch (err) {
      setError(
        err?.response?.data?.message ||
          err?.message ||
          'Unable to fetch suggestion.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-10">
      <AnimatedSection>
        <div className="space-y-3">
          <p className="text-green-primary uppercase tracking-widest text-xs">
            Smart Crop Lab
          </p>
          <h2 className="text-3xl font-display">Precision crop insights</h2>
          <p className="text-text-secondary max-w-2xl">
            Feed soil nutrients and climate signals to generate the most viable
            crop recommendation tailored for your acreage.
          </p>
        </div>
      </AnimatedSection>

      <AnimatedSection delay={0.1}>
        <GlassCard className="relative">
          {isLoading && <LoadingOverlay label="Running agronomic model..." />}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="grid md:grid-cols-3 gap-4">
              <InputField
                label="Nitrogen (N)"
                name="N"
                type="number"
                min="0"
                step="0.1"
                value={formValues.N}
                onChange={handleChange}
              />
              <InputField
                label="Phosphorus (P)"
                name="P"
                type="number"
                min="0"
                step="0.1"
                value={formValues.P}
                onChange={handleChange}
              />
              <InputField
                label="Potassium (K)"
                name="K"
                type="number"
                min="0"
                step="0.1"
                value={formValues.K}
                onChange={handleChange}
              />
              <InputField
                label="Temperature"
                suffix="°C"
                name="temperature"
                type="number"
                step="0.1"
                value={formValues.temperature}
                onChange={handleChange}
              />
              <InputField
                label="Humidity"
                suffix="%"
                name="humidity"
                type="number"
                step="0.1"
                value={formValues.humidity}
                onChange={handleChange}
              />
              <InputField
                label="Soil pH"
                name="ph"
                type="number"
                step="0.1"
                value={formValues.ph}
                onChange={handleChange}
              />
              <InputField
                label="Rainfall"
                suffix="mm"
                name="rainfall"
                type="number"
                step="0.1"
                value={formValues.rainfall}
                onChange={handleChange}
              />
            </div>
            {error && (
              <p className="text-sm text-red-500 bg-red-500/5 px-4 py-2 rounded-xl border border-red-200">
                {error}
              </p>
            )}
            <motion.button
              type="submit"
              className="w-full md:w-auto px-8 py-3 rounded-2xl bg-green text-olive-text font-semibold shadow-glow hover:bg-green-light hover:-translate-y-0.5 transition text-lime-950"
              whileTap={{ scale: 0.98 }}
            >
              Get Recommendation
            </motion.button>
          </form>
        </GlassCard>
      </AnimatedSection>

      {result && (
        <AnimatedSection delay={0.2}>
          <ResultCard
            title={result?.predicted_crop || 'Suggested Crop'}
            // badge={result?.confidence && `Confidence: ${result.confidence}%`}
            description={
              result?.insight ||
              'Ensure balanced nutrients and moisture for optimal yield.'
            }
            extra={
              (result?.tips && (
                <ul className="text-text-secondary text-sm list-disc pl-5 space-y-1">
                  {result.tips.map((tip) => (
                    <li key={tip}>{tip}</li>
                  ))}
                </ul>
              )) ||
              (result?.candidates?.length > 0 && (
                <div className="space-y-2 text-sm text-text-secondary">
                  <p className="text-xs uppercase tracking-wide text-text-secondary/80">
                    Top candidates
                  </p>
                  <ul className="space-y-1">
                    {result.candidates.map((c) => (
                      <li
                        key={c.crop}
                        className="flex items-center justify-between rounded-xl bg-green-light px-3 py-2"
                      >
                        <span>{c.crop}</span>
                        {/* {c.confidence != null && ( */}
                          {/*  <span className="text-text-olive font-medium"> */}
                          {/*    {c.confidence}% */}
                          {/*  </span> */}
                        {/* )} */}
                      </li>
                    ))}
                  </ul>
                </div>
              ))
            }
          />
        </AnimatedSection>
      )}
    </div>
  );
};

export default SmartCrop;

