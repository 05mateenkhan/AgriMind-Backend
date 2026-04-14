import os
import base64
import io
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import pandas as pd
from pydantic import BaseModel

from utils.helpers import safe_float, save_uploaded_image, predict_disease
from models import PredictPipeline, CustomData

router = APIRouter()

# Load data at module level
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), 'data')

disease_info = pd.read_csv(os.path.join(data_dir, "disease_info.csv"), encoding='cp1252')
supplement_info = pd.read_csv(os.path.join(data_dir, 'supplement_info.csv'), encoding='cp1252')

UPLOAD_FOLDER = os.path.join(os.path.dirname(script_dir), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Pydantic models for request/response
class CropPredictionInput(BaseModel):
    n: float
    p: float
    k: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class CropPredictionResult(BaseModel):
    crop: str
    confidence: Optional[float] = None


class InputData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class PredictionResponse(BaseModel):
    status: str
    input_data: Optional[InputData] = None
    prediction: Optional[List[CropPredictionResult]] = None
    message: Optional[str] = None


@router.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """Handle image upload and return prediction results as JSON."""
    try:
        # Save uploaded file
        contents = await file.read()
        file_obj = io.BytesIO(contents)
        file_path = save_uploaded_image(file_obj, file.filename, UPLOAD_FOLDER)

        # Verify image
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception:
            try:
                os.remove(file_path)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

        # Get prediction
        pred = predict_disease(file_path)

        result = {
            "disease_name": disease_info['disease_name'][pred],
            "description": disease_info['description'][pred],
            "possible_steps": disease_info['Possible Steps'][pred],
            "image_url": disease_info['image_url'][pred],
            "supplement": {
                "name": supplement_info['supplement name'][pred],
                "image_url": supplement_info['supplement image'][pred],
                "buy_link": supplement_info['buy link'][pred]
            }
        }

        return result
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Server error processing image", "detail": str(e)}
        )


@router.post("/api/predictdata", response_model=PredictionResponse)
async def predict_datapoint(
    n: float = Form(...),
    p: float = Form(...),
    k: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    ph: float = Form(...),
    rainfall: float = Form(...)
):
    """Handle crop prediction with soil/weather data."""
    try:
        data = CustomData(n, p, k, temperature, humidity, ph, rainfall)
        data_df = CustomData.get_data_as_frame(data)
        predict_pipeline = PredictPipeline()

        if hasattr(predict_pipeline.model, "predict_proba"):
            probs = predict_pipeline.model.predict_proba(data_df)[0]
            classes = predict_pipeline.model.classes_
            top3_indices = probs.argsort()[-3:][::-1]
            top3_predictions = [
                {"crop": classes[i], "confidence": round(float(probs[i]) * 100, 2)}
                for i in top3_indices
            ]
        else:
            result = predict_pipeline.predict(data_df)
            top3_predictions = [{"crop": result[0], "confidence": None}]

        return {
            "status": "success",
            "input_data": {
                "N": data.n, "P": data.p, "K": data.k,
                "temperature": data.temperature, "humidity": data.humidity,
                "ph": data.ph, "rainfall": data.rainfall
            },
            "prediction": top3_predictions
        }
    except Exception as e:
        return PredictionResponse(
            status="error",
            message=str(e)
        )


@router.get("/api/market")
async def get_market():
    """Return all supplements and related info."""
    supplements = []
    for i in range(len(supplement_info)):
        supplements.append({
            "supplement_name": supplement_info['supplement name'][i],
            "disease_name": disease_info['disease_name'][i],
            "image_url": supplement_info['supplement image'][i],
            "buy_link": supplement_info['buy link'][i]
        })
    return {"market": supplements}