from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from skimage.color import rgb2lab
import colorsys

app = Flask(__name__)
CORS(app)  # ✅ Allow frontend JS to access

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def extract_features(rgb):
    rgb_scaled = np.array(rgb) / 255.0
    hsv = colorsys.rgb_to_hsv(*rgb_scaled)
    lab = rgb2lab(np.reshape(rgb_scaled, (1, 1, 3)))

    b_lab = lab[0, 0, 2]
    log_hue = np.log(hsv[0] + 1e-5)
    log_green = np.log(rgb_scaled[1] + 1e-5)

    return np.array([[b_lab, log_hue, log_green]])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    rgb = data["rgb"]
    features = extract_features(rgb)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    # ✅ Ensure return type is JSON serializable
    return jsonify({"prediction": int(prediction[0])})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

