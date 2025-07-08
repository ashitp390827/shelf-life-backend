from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from skimage.color import rgb2lab
import colorsys
import os

app = Flask(__name__)
CORS(app)

# ✅ Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ✅ Constants for transformations (you provided)
GREEN_MIN = 35.17360876
GREEN_MAX = 184.3057141
HUE_MIN = 12.92046354
HUE_MAX = 56.73791461

# ✅ Feature extraction function with corrected transformations
def extract_features(rgb):
    rgb_scaled = np.array(rgb) / 255.0  # Normalize to [0,1]
    
    hsv = colorsys.rgb_to_hsv(*rgb_scaled)
    lab = rgb2lab(np.reshape(rgb_scaled, (1, 1, 3)))

    b_lab = lab[0, 0, 2]  # LAB B channel

    # Custom log transformation for hue
    hue = hsv[0] * 360  # Hue in degrees
    hue_transformed = np.log1p((hue - HUE_MIN) / (HUE_MAX - HUE_MIN))

    # Custom log transformation for greenness (G channel)
    greenness = rgb_scaled[1] * 255  # Green channel in [0,255]
    green_transformed = np.log1p((greenness - GREEN_MIN) / (GREEN_MAX - GREEN_MIN))
    print(f"Extracted features - B: {b_lab}, Hue: {hue_transformed}, Green: {green_transformed}")
    return np.array([[b_lab, hue_transformed, green_transformed]])

# ✅ Health check route
@app.route("/", methods=["GET"])
def home():
    return "✅ Shelf-life predictor backend is running!", 200

# ✅ Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "rgb" not in data:
            return jsonify({"error": "Missing 'rgb' in request"}), 400

        rgb = data["rgb"]
        features = extract_features(rgb)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        print(f"Extracted features - B: {b_lab}, Hue: {hue_transformed}, Green: {green_transformed}")
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": "Internal server error"}), 500

# ✅ Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
