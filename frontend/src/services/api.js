import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:5000',
  timeout: 20000,
});

// Smart Crop Recommendation – send as form-data & normalize backend response
export const predictCrop = async (payload) => {
  const formData = new FormData();
  // Backend expects lowercase keys in request.form
  formData.append('n', payload.N ?? payload.n ?? '');
  formData.append('p', payload.P ?? payload.p ?? '');
  formData.append('k', payload.K ?? payload.k ?? '');
  formData.append('temperature', payload.temperature ?? '');
  formData.append('humidity', payload.humidity ?? '');
  formData.append('ph', payload.ph ?? '');
  formData.append('rainfall', payload.rainfall ?? '');

  const { data } = await api.post('/api/predictdata', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  // Expected backend shape:
  // {
  //   status: "success",
  //   input_data: { ... },
  //   prediction: [{ crop, confidence }, ...]
  // }
  const candidates = Array.isArray(data?.prediction) ? data.prediction : [];
  const top = candidates[0] || {};

  return {
    raw: data,
    predicted_crop: top.crop,
    confidence: top.confidence,
    candidates,
  };
};

export const predictDisease = async (file) => {
  const formData = new FormData();
  formData.append('image', file);
  const { data } = await api.post('/api/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
};

export default api;

