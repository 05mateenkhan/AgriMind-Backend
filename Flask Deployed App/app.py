import os
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from flask_cors import CORS  # Allow React frontend to connect
from predict_pipeline import CustomData
from predict_pipeline import PredictPipeline

# Load data and model
script_dir = os.path.dirname(os.path.abspath(__file__))
disease_info = pd.read_csv(os.path.join(script_dir, "disease_info.csv"),
    encoding='cp1252'
)
supplement_info = pd.read_csv(os.path.join(script_dir, 'supplement_info.csv'), encoding='cp1252')

model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu')))
model.eval()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend (important!)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_disease(image_path):
    """Run prediction using CNN model"""
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).view((-1, 3, 224, 224))
    with torch.no_grad():
        output = model(input_data)
    output = output.numpy()
    index = np.argmax(output)
    return int(index)

@app.route('/')
def health_check():
    return jsonify({"message": "Flask backend is running"}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle image upload and return prediction results as JSON"""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image']
    filename = image.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(file_path)

    pred = predict_disease(file_path)

    # Gather data
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




@app.route('/api/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return jsonify({"error": "No data uploaded"}), 400
    else:  
        try:
            data = CustomData(
                n=request.form.get('n'), 
                p=request.form.get('p'),
                k=request.form.get('k'),
                temperature=request.form.get('temperature'),
                humidity=request.form.get('humidity'),
                ph=float(request.form.get('ph')),
                rainfall=float(request.form.get('rainfall'))
            ) 
            data_df = CustomData.get_data_as_frame(data)
            predict_pipeline = PredictPipeline()


            if hasattr(predict_pipeline.model, "predict_proba"):
                probs = predict_pipeline.model.predict_proba(data_df)[0]
                classes = predict_pipeline.model.classes_

                # Get top 3 indices by probability
                top3_indices = probs.argsort()[-3:][::-1]
                top3_predictions = [
                    {"crop": classes[i], "confidence": round(float(probs[i]) * 100, 2)}
                    for i in top3_indices
                ]
            else:
                # ---- Option 2: If model only supports predict() ----
                # Fallback to single output (repeat or dummy list)
                result = predict_pipeline.predict(data_df)
                top3_predictions = [{"crop": result[0], "confidence": None}]

            return jsonify({
                    "status": "success",
                    "input_data": {
                        "N": data.n,
                        "P": data.p,
                        "K": data.k,
                        "temperature": data.temperature,
                        "humidity": data.humidity,
                        "ph": data.ph,
                        "rainfall": data.rainfall
                    },
                    "prediction": top3_predictions
                }), 200
            

        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    





@app.route('/api/market', methods=['GET'])
def get_market():
    """Return all supplements and related info"""
    supplements = []
    for i in range(len(supplement_info)):
        supplements.append({
            "supplement_name": supplement_info['supplement name'][i],
            "disease_name": disease_info['disease_name'][i],
            "image_url": supplement_info['supplement image'][i],
            "buy_link": supplement_info['buy link'][i]
        })
    return jsonify({"market": supplements}), 200

if __name__ == '__main__':
    app.run(debug=True)
