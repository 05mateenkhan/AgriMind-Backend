import os
import base64
import io
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
from werkzeug.utils import secure_filename

# Get the backend root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model is loaded at module level
model = None


def load_model():
    """Lazy load the CNN model"""
    global model
    if model is None:
        from models.CNN import CNN
        model = CNN(39)
        model_path = os.path.join(BASE_DIR, 'plant_disease_model_1_latest.pt')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    return model


def safe_float(value, name=None, default=None):
    """Convert value to float, raise ValueError with helpful message if missing/invalid."""
    if value is None or value == "":
        if default is not None:
            return float(default)
        raise ValueError(f"missing numeric field: {name}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid numeric value for {name}: {value!r}")


def save_uploaded_image(fileobj, filename, upload_folder):
    """Save a FileStorage or BytesIO to the uploads folder and return the saved path."""
    filename = secure_filename(filename) if filename else 'upload.jpg'
    file_path = os.path.join(upload_folder, filename)
    if hasattr(fileobj, 'save'):
        fileobj.save(file_path)
    else:
        with open(file_path, 'wb') as f:
            try:
                f.write(fileobj.getbuffer())
            except Exception:
                fileobj.seek(0)
                f.write(fileobj.read())
    return file_path


def predict_disease(image_path):
    """Run prediction using CNN model"""
    m = load_model()
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).view((-1, 3, 224, 224))
    with torch.no_grad():
        output = m(input_data)
    output = output.numpy()
    index = np.argmax(output)
    return int(index)