import os
import sys
import pandas as pd
import numpy as np
from utils.exception import CustomException
import dill

# Get the backend root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PredictPipeline:
    def __init__(self):
        """Load the model once when creating the pipeline."""
        try:
            model_path = os.path.join(BASE_DIR, 'RandomForest.pkl')
            with open(model_path, 'rb') as file_obj:
                self.model = dill.load(file_obj)
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features):
        """
        Predict top 3 most suitable crops for the given soil/weather features.
        """
        try:
            # ✅ use self.model directly
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)

                # Get top 3 class indices (highest probabilities)
                top3_indices = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]

                # Map indices to actual crop names (class labels)
                top3_crops = [[self.model.classes_[i] for i in indices] for indices in top3_indices]
                return top3_crops

            else:
                # fallback to single prediction
                prediction = self.model.predict(features)
                return [[prediction[0]]]

        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self, n, p, k, temperature, humidity, ph, rainfall):
        self.n = n
        self.p = p
        self.k = k
        self.temperature = temperature
        self.humidity = humidity
        self.ph = ph
        self.rainfall = rainfall

    def get_data_as_frame(self):
        """
        Convert the custom data into a DataFrame for model prediction.
        """
        try:
            data_dict = {
                "N": [float(self.n)],
                "P": [float(self.p)],
                "K": [float(self.k)],
                "temperature": [float(self.temperature)],
                "humidity": [float(self.humidity)],
                "ph": [float(self.ph)],
                "rainfall": [float(self.rainfall)],
            }
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)