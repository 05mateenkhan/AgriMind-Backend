import os
import base64
import io
from flask import Blueprint, request, jsonify
from PIL import Image
import pandas as pd

from utils.helpers import safe_float, save_uploaded_image, predict_disease
from models import PredictPipeline, CustomData

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Load data at module level
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(os.path.dirname(script_dir), 'data')

disease_info = pd.read_csv(os.path.join(data_dir, "disease_info.csv"), encoding='cp1252')
supplement_info = pd.read_csv(os.path.join(data_dir, 'supplement_info.csv'), encoding='cp1252')

UPLOAD_FOLDER = os.path.join(os.path.dirname(script_dir), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@api_bp.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction results as JSON."""
    file_obj = None
    filename = None

    if 'image' in request.files:
        file_obj = request.files.get('image')
        filename = getattr(file_obj, 'filename', None)

    if file_obj is None and request.is_json:
        payload = request.get_json(silent=True) or {}
        img_b64 = payload.get('image')
        if img_b64:
            if isinstance(img_b64, str) and img_b64.startswith('data:'):
                img_b64 = img_b64.split(',', 1)[1]
            try:
                img_bytes = base64.b64decode(img_b64)
                file_obj = io.BytesIO(img_bytes)
                filename = payload.get('filename', 'upload.jpg')
            except Exception as e:
                return jsonify({"error": "Invalid base64 image data"}), 400

    if file_obj is None:
        return jsonify({"error": "No image provided"}), 400

    try:
        file_path = save_uploaded_image(file_obj, filename, UPLOAD_FOLDER)

        try:
            img = Image.open(file_path)
            img.verify()
        except Exception:
            try:
                os.remove(file_path)
            except Exception:
                pass
            return jsonify({"error": "Uploaded file is not a valid image"}), 400

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

        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": "Server error processing image", "detail": str(e)}), 500


@api_bp.route('/predictdata', methods=['POST'])
def predict_datapoint():
    """Handle crop prediction with soil/weather data."""
    payload = {}
    if request.is_json:
        payload = request.get_json(silent=True) or {}
    else:
        payload = request.form.to_dict()

    try:
        n = safe_float(payload.get('n'), 'n')
        p = safe_float(payload.get('p'), 'p')
        k = safe_float(payload.get('k'), 'k')
        temperature = safe_float(payload.get('temperature'), 'temperature')
        humidity = safe_float(payload.get('humidity'), 'humidity')
        ph = safe_float(payload.get('ph'), 'ph')
        rainfall = safe_float(payload.get('rainfall'), 'rainfall')

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

        return jsonify({
            "status": "success",
            "input_data": {
                "N": data.n, "P": data.p, "K": data.k,
                "temperature": data.temperature, "humidity": data.humidity,
                "ph": data.ph, "rainfall": data.rainfall
            },
            "prediction": top3_predictions
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route('/market', methods=['GET'])
def get_market():
    """Return all supplements and related info."""
    supplements = []
    for i in range(len(supplement_info)):
        supplements.append({
            "supplement_name": supplement_info['supplement name'][i],
            "disease_name": disease_info['disease_name'][i],
            "image_url": supplement_info['supplement image'][i],
            "buy_link": supplement_info['buy link'][i]
        })
    return jsonify({"market": supplements}), 200