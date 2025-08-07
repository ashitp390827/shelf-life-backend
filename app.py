import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from skimage.color import rgb2lab
import colorsys
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("neural_network_model.joblib")
scaler = joblib.load("standard_scaler.joblib")

def extract_all_features(rgb):
    rgb = np.array(rgb)
    rgb_scaled = rgb / 255.0

    r, g, b = rgb
    r_scaled, g_scaled, b_scaled = rgb_scaled

    # HSV
    h, s, v = colorsys.rgb_to_hsv(r_scaled, g_scaled, b_scaled)
    hue_deg = h * 360

    # LAB
    lab = rgb2lab(np.reshape(rgb_scaled, (1, 1, 3)))
    L, a, b_lab = lab[0, 0, :]

    # Raw features
    features = {
        "Redness": r,
        "Greeness": g,
        "Blueness": b,
        "L* lab": L,
        "a* lab": a,
        "b* lab": b_lab,
        "Hue": hue_deg,
        "Saturation": s,
        "Value": v,

        # Log-transformed features
        "log-Redness": np.log1p(r),
        "log-Greeness": np.log1p(g),
        "log-Blueness": np.log1p(b),
        "log-L* lab": np.log1p(L),
        "log-a* lab": np.log1p(128 + a),  # a* in LAB can be negative
        "log-b* lab": np.log1p(128 + b_lab),
        "log-Hue": np.log1p(hue_deg),
        "log-Saturation": np.log1p(s),
        "log-Value": np.log1p(v)
    }

    return features

@app.route("/", methods=["GET"])
def home():
    return "✅ Shelf-life predictor backend is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "control_rgb" not in data or "indicator_rgb" not in data:
            return jsonify({"error": "Missing RGB values in request"}), 400

        control_rgb = np.array(data["control_rgb"])
        indicator_rgb = np.array(data["indicator_rgb"])
        std_rgb = np.array([184, 159, 8])

        correction = std_rgb - control_rgb
        corrected_rgb = np.clip(indicator_rgb + correction, 0, 255)

        print(f"Control RGB: {control_rgb}, Indicator RGB: {indicator_rgb}")
        print(f"Correction: {correction}, Corrected RGB: {corrected_rgb}")

        feature_dict = extract_all_features(corrected_rgb)

        # Select only model-required features for prediction
        model_features = np.array([[feature_dict["b* lab"],
                                    feature_dict["log-Hue"],
                                    feature_dict["Saturation"]]])

        model_features_df = pd.DataFrame([{
            "b* lab": feature_dict["b* lab"],
            "log-Hue": feature_dict["log-Hue"],
            "Saturation": feature_dict["Saturation"]
        }])

        features_scaled = scaler.transform(model_features_df)
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)

        predicted_class = int(prediction[0])
        confidence = float(np.max(probabilities))
        print(feature_dict)
        return jsonify({
            "prediction": int(predicted_class),
            "confidence": float(confidence),
            "corrected_rgb": [int(x) for x in corrected_rgb.tolist()],

        })


    except Exception as e:
        print("❌ Error during prediction:", str(e))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
