from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)

MODEL_DIR = "model"
MODEL_PATH = os.getenv("MODEL_PATH", f"{MODEL_DIR}/vgg16_rgb_final_model.h5")

model = load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

def prepare_image(img_file):
    img = image.load_img(BytesIO(img_file.read()), target_size=(128, 128), color_mode="rgb")
    img_array = image.img_to_array(img).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0), img_array  # Normalized + raw

def is_valid_rgb_ndvi(image_array):
    if image_array.shape != (256, 256, 3):  # ✅ Adjusted shape check
        return False

    mean_val = np.mean(image_array * 255.0)
    std_val = np.std(image_array * 255.0)

    if not (100 <= mean_val <= 160):
        return False

    if std_val < 100 or std_val > 115:
        return False

    red_mean = np.mean(image_array[:, :, 0] * 255.0)
    green_mean = np.mean(image_array[:, :, 1] * 255.0)
    blue_mean = np.mean(image_array[:, :, 2] * 255.0)

    if not (blue_mean > green_mean > red_mean):
        return False

    if not (0 <= red_mean <= 50 and 60 <= green_mean <= 210 and 215 <= blue_mean <= 255):
        return False

    return True

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_array, raw_array = prepare_image(img_file)

    # ✅ Validate NDVI image
    if not is_valid_rgb_ndvi(raw_array):
        return jsonify({
            "class": "Invalid",
            "confidence": 0,
            "message": "⚠️ The uploaded image does not match the expected NDVI RGB pattern for rice crop leaves."
        }), 200

    pred_prob = float(model.predict(img_array)[0][0])
    threshold = 0.5
    predicted_class = "healthy" if pred_prob >= threshold else "diseased"
    confidence = pred_prob if pred_prob >= threshold else 1 - pred_prob

    return jsonify({
        "class": predicted_class,
        "confidence": round(confidence, 4)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
