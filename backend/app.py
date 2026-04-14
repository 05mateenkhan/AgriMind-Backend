import os
from flask import Flask, jsonify
from flask_cors import CORS
import logging

from routes.api import api_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(api_bp)

# Logging
logging.basicConfig(level=logging.INFO)


@app.route('/')
def health_check():
    return jsonify({"message": "Flask backend is running"}), 200


if __name__ == '__main__':
    app.run(debug=True)