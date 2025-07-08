from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from skimage.color import rgb2lab
import colorsys
import os

app = Flask(__name__)
CORS(app)  # ✅ Allow frontend JS to access

# ✅ Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ✅ Feature extraction function
def extract_features(rgb):
    rgb_scaled = np.array(rgb) / 255.0
    hsv = colorsys.rgb_to_hsv(*rgb_scaled)
    lab = rgb2lab(np.reshape(rgb_scaled, (1, 1, 3)))

    b_lab = lab[0, 0, 2]
    log_hue = np.log(hsv[0] + 1e-5)
    log_green = np.log(rgb_scaled[1] + 1e-5)

    return np.array([[b_lab, log_hue, log_green]])

# ✅ Health check route (for browser or Render checks)

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

        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": "Internal server error"}), 500

# ✅ Run server (Render-compatible)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render sets PORT env var
    app.run(host="0.0.0.0", port=port)
