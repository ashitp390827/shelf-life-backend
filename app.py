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
model = joblib.load("logistic_regression_model.joblib")

scaler = joblib.load("standard_scaler.joblib")

# ✅ Constants for transformations
GREEN_MIN = 35.17360876
GREEN_MAX = 184.3057141
HUE_MIN = 12.92046354
HUE_MAX = 56.73791461

# ✅ Feature extraction function
def extract_features(rgb):
    rgb_scaled = np.array(rgb) / 255.0  # Normalize to [0,1]

    hsv = colorsys.rgb_to_hsv(*rgb_scaled)
    lab = rgb2lab(np.reshape(rgb_scaled, (1, 1, 3)))

    b_lab = lab[0, 0, 2]  # LAB B channel

    # Transform hue and green channel using log
    hue = hsv[0] * 360  # convert hue to degrees
    hue_transformed = np.log1p(hue)

    greenness = rgb_scaled[1] * 255  # green in 0–255
    green_transformed = np.log1p(greenness)

    print(f"Extracted features - B: {b_lab}, log-Hue: {hue_transformed}, log-Green: {green_transformed}")
    return np.array([[b_lab, hue_transformed, greenness]])

# ✅ Health check
@app.route("/", methods=["GET"])
def home():
    return "✅ Shelf-life predictor backend is running!", 200

# ✅ Prediction with confidence
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "control_rgb" not in data or "indicator_rgb" not in data:
            return jsonify({"error": "Missing RGB values in request"}), 400

        control_rgb = np.array(data["control_rgb"])
        indicator_rgb = np.array(data["indicator_rgb"])
        std_rgb = np.array([184, 159, 8])  # your defined standard RGB

        # ✅ Additive correction
        correction = std_rgb - control_rgb
        corrected_rgb = np.clip(indicator_rgb + correction, 0, 255)

        print(f"Control RGB: {control_rgb}, Indicator RGB: {indicator_rgb}")
        print(f"Correction: {correction}, Corrected RGB: {corrected_rgb}")

        # ✅ Feature extraction and prediction
        features = extract_features(corrected_rgb)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)

        predicted_class = int(prediction[0])
        confidence = float(np.max(probabilities))

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence,
            "corrected_rgb": corrected_rgb.tolist()
        })

    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": "Internal server error"}), 500


# ✅ Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
