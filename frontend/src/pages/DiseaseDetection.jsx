import { useState } from 'react';
import GlassCard from '../components/GlassCard.jsx';
import FileUpload from '../components/FileUpload.jsx';
import ResultCard from '../components/ResultCard.jsx';
import LoadingOverlay from '../components/LoadingOverlay.jsx';
import AnimatedSection from '../components/AnimatedSection.jsx';
import { predictDisease } from '../services/api.js';

const DiseaseDetection = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleFile = (event) => {
    const selected = event.target.files?.[0];
    if (!selected) return;
    setError('');
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      setError('Please upload a field photo to continue.');
      return;
    }
    setIsLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await predictDisease(file);
      setResult(data);
    } catch (err) {
      setError(err?.response?.data?.message || 'Detection failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto space-y-10">
      <AnimatedSection>
        <div className="space-y-3">
          <p className="text-green-primary uppercase tracking-widest text-xs">
            Disease Vision
          </p>
          <h2 className="text-3xl font-display">Diagnose plant stress early</h2>
          <p className="text-text-secondary max-w-2xl">
            Upload a leaf photo and let the CNN engine classify disease, report
            confidence, and deliver mitigation strategies instantly.
          </p>
        </div>
      </AnimatedSection>

      <AnimatedSection delay={0.1}>
        <GlassCard className="relative">
          {isLoading && <LoadingOverlay label="Scanning plant health..." />}
          <form onSubmit={handleSubmit} className="space-y-6">
            <FileUpload
              onChange={handleFile}
              preview={preview}
              isUploading={isLoading}
            />
            {error && (
              <p className="text-sm text-red-500 bg-red-500/5 px-4 py-2 rounded-xl border border-red-200">
                {error}
              </p>
            )}
            <button
              type="submit"
              className="w-full md:w-auto px-8 py-3 rounded-2xl bg-green-accent text-olive-text font-semibold shadow-glow hover:bg-green-light hover:-translate-y-0.5 transition text-lime-800"
            >
              Detect Disease
            </button>
          </form>
        </GlassCard>
      </AnimatedSection>

      {result && (
        <AnimatedSection delay={0.2}>
          <ResultCard
            title={result?.disease_name || 'Detected disease'}
            badge={null}
            description={
              result?.description ||
              'Maintain balanced irrigation and monitor canopy for early signs.'
            }
            extra={
              <div className="space-y-4 text-sm text-muted-text">
                {/* Disease reference image from backend */}
                {result?.image_url && (
                  <div className="rounded-2xl overflow-hidden border border-border-cream bg-cream-card">
                    {/* <img
                      src={result.image_url}
                      alt={result.disease_name || 'Disease reference'}
                      className="w-full h-56 object-cover"
                    /> */}
                  </div>
                )}

                {/* Possible steps / management guidance */}
                {result?.possible_steps && (
                  <div className="space-y-1">
                    <p className="text-charcoal font-medium">
                      Suggested next steps
                    </p>
                    <p className="text-text-secondary leading-relaxed">
                      {result.possible_steps}
                    </p>
                  </div>
                )}

                {/* Supplement information block */}
                {result?.supplement && (
                  <div className="mt-2 rounded-2xl border border-border-cream bg-cream-card p-4 flex flex-col md:flex-row gap-4 items-start">
                    {result.supplement.image_url && (
                      <div className="w-24 h-24 rounded-xl overflow-hidden border border-border-cream bg-white flex-shrink-0">
                        <img
                          src={result.supplement.image_url}
                          alt={result.supplement.name || 'Supplement'}
                          className="w-full h-full object-cover"
                        />
                      </div>
                    )}
                    <div className="space-y-2">
                      <p className="text-xs uppercase tracking-wide text-green-primary">
                        Recommended supplement
                      </p>
                      {result.supplement.name && (
                        <p className="text-charcoal font-semibold">
                          {result.supplement.name}
                        </p>
                      )}
                      {result.supplement.buy_link && (
                        <a
                          href={result.supplement.buy_link}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-2 text-xs font-medium text-lime-800 bg-yellow-corn/40 px-3 py-1.5 rounded-full hover:bg-yellow-corn/60 transition"
                        >
                          View product / Buy online
                          <span aria-hidden="true">↗</span>
                        </a>
                      )}
                    </div>
                  </div>
                )}
              </div>
            }
          />
        </AnimatedSection>
      )}
    </div>
  );
};

export default DiseaseDetection;

