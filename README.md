# AgriMind

An AI-powered agricultural intelligence platform that provides smart crop recommendations and plant disease detection using machine learning.

## Features

### 1. Smart Crop Recommendation
- Analyzes soil nutrients (N, P, K), weather data (temperature, humidity, rainfall), and pH levels
- Uses Random Forest classifier to predict the most suitable crops for given conditions
- Returns top 3 crop recommendations with confidence scores

### 2. Disease Vision AI
- Upload plant leaf images for disease detection
- Uses a Convolutional Neural Network (CNN) trained on 39 plant disease classes
- Provides disease name, description, possible steps, and recommended supplements

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React + Vite + Tailwind CSS + Framer Motion |
| Backend | Flask (Python) + Flask-CORS |
| ML Models | PyTorch (CNN), scikit-learn (Random Forest) |
| Data | Pandas, NumPy |

## Project Structure

```
AgriMind/
├── backend/                    # Flask API server
│   ├── app.py                  # Main entry point
│   ├── requirements.txt        # Python dependencies
│   ├── data/                   # CSV data files
│   │   ├── disease_info.csv    # Disease descriptions & info
│   │   └── supplement_info.csv # Recommended supplements
│   ├── models/                 # ML models
│   │   ├── CNN.py              # PyTorch CNN model
│   │   └── predict_pipeline.py # Random Forest pipeline
│   ├── utils/                  # Utilities
│   │   ├── exception.py        # Custom exception handler
│   │   └── helpers.py          # Helper functions
│   ├── routes/                # API routes
│   │   └── api.py              # API endpoints
│   ├── static/uploads/         # Uploaded images
│   ├── RandomForest.pkl        # Trained crop prediction model
│   └── plant_disease_model_1_latest.pt  # Trained disease detection model
│
├── frontend/                   # React application
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   ├── pages/             # Page components
│   │   │   ├── Home.jsx        # Landing page
│   │   │   ├── SmartCrop.jsx   # Crop recommendation page
│   │   │   └── DiseaseDetection.jsx  # Disease detection page
│   │   ├── services/
│   │   │   └── api.js          # API client
│   │   └── App.jsx             # Main app component
│   └── package.json
│
├── test_images/                # Test images for both features
├── .gitignore
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/api/predict` | POST | Image-based disease detection |
| `/api/predictdata` | POST | Crop recommendation from soil/weather data |
| `/api/market` | GET | List all supplements |

### Request/Response Examples

#### Disease Detection
```bash
# Multipart form data
curl -X POST http://localhost:5000/api/predict \
  -F "image=@leaf.jpg"

# Response
{
  "disease_name": "Tomato___Late_blight",
  "description": "Late blight is a disease...",
  "possible_steps": "Apply fungicide...",
  "image_url": "...",
  "supplement": {
    "name": "Copper Fungicide",
    "image_url": "...",
    "buy_link": "..."
  }
}
```

#### Crop Recommendation
```bash
# Form data
curl -X POST http://localhost:5000/api/predictdata \
  -d "n=90&p=42&k=43&temperature=20&humidity=80&ph=6.5&rainfall=200"

# Response
{
  "status": "success",
  "input_data": {
    "N": 90, "P": 42, "K": 43,
    "temperature": 20, "humidity": 80,
    "ph": 6.5, "rainfall": 200
  },
  "prediction": [
    {"crop": "coffee", "confidence": 92.5},
    {"crop": "tea", "confidence": 5.2},
    {"crop": "banana", "confidence": 2.1}
  ]
}
```

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+

### Backend Setup

```bash
cd backend

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The backend runs on `http://127.0.0.1:5000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend runs on `http://localhost:5173`

## Disease Classes

The CNN model can detect 39 plant disease classes including:

- Apple (Apple scab, Black rot, Cedar apple rust, healthy)
- Blueberry (healthy)
- Cherry (Powdery mildew, healthy)
- Corn (Cercospora leaf spot, Common rust, Northern Leaf Blight, healthy)
- Grape (Black rot, Esca, Leaf blight, healthy)
- Orange (Haunglongbing)
- Peach (Bacterial spot, healthy)
- Pepper bell (Bacterial spot, healthy)
- Potato (Early blight, Late blight, healthy)
- Raspberry (healthy)
- Soybean (healthy)
- Squash (Powdery mildew)
- Strawberry (Leaf scorch, healthy)
- Tomato (Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Tomato mosaic virus, healthy)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment | `development` |
| `FLASK_DEBUG` | Enable debug mode | `1` |

## License

MIT License