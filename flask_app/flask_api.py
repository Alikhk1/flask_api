import os
import gdown
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow.lite as tflite
import cv2
import joblib

app = Flask(__name__)
CORS(app)

# Paths for Railway deployment
MODEL_PATH = "/app/flask_app/model.tflite"
SCALER_PATH = "/app/flask_app/scaler.pkl"

# Google Drive File ID
MODEL_FILE_ID = "1-0E4AQHqYJdJxqx0MSWb1uNVxtDZh78k"  # Your model file ID from Drive

def download_model():
    """Download the model from Google Drive if not found locally."""
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("✅ Model already exists.")

# Ensure the model is downloaded
download_model()

# Load the TFLite model
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    interpreter = None

# Load the saved scaler
try:
    scaler = joblib.load(SCALER_PATH)
    print("✅ Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    scaler = None

# Define measurement names
measurement_names = ["Arm Length", "Shoulder Width", "Chest", "Waist", "Hip", "Neck"]

# Define size ranges
size_ranges = {
    "Small": {"Chest": (32, 36), "Waist": (28, 32), "Hip": (32, 36), "Shoulder Width": (14, 16), "Arm Length": (22, 24), "Neck": (14, 15)},
    "Medium": {"Chest": (37, 40), "Waist": (33, 36), "Hip": (37, 40), "Shoulder Width": (16.5, 18), "Arm Length": (24.5, 26), "Neck": (15.5, 16.5)},
    "Large": {"Chest": (41, 44), "Waist": (37, 40), "Hip": (41, 44), "Shoulder Width": (18.5, 20), "Arm Length": (26.5, 28), "Neck": (17, 18)}
}

def determine_size(measurements):
    """Determines clothing size (S, M, L) based on predicted measurements."""
    size_counts = {"Small": 0, "Medium": 0, "Large": 0}

    for i, value in enumerate(measurements):
        for size, ranges in size_ranges.items():
            min_val, max_val = ranges[measurement_names[i]]
            if min_val <= value <= max_val:
                size_counts[size] += 1

    return max(size_counts, key=size_counts.get) if max(size_counts.values()) > 0 else "Unknown"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    if interpreter is None:
        return jsonify({"error": "Model not loaded properly."}), 500
    
    if scaler is None:
        return jsonify({"error": "Scaler not loaded properly."}), 500

    image = request.files["image"]
    image_bytes = image.read()

    try:
        # Preprocess image
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))  # Ensure it matches model input size
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Convert predictions back to real measurements
        actual_measurements = scaler.inverse_transform([predictions])[0]

        # Format response
        response = {measurement_names[i]: f"{actual_measurements[i]:.2f}" for i in range(len(actual_measurements))}
        response["Recommended Size"] = determine_size(actual_measurements)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
