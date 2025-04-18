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

model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

def prepare_image(img_file):
    img = image.load_img(BytesIO(img_file.read()), target_size=(128, 128), color_mode="rgb")
    img_array = image.img_to_array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        return jsonify({"error": "Model not loaded. Upload it via /upload-model first."}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_array = prepare_image(img_file)

    # ✅ Apply threshold-based binary classification
    pred_prob = float(model.predict(img_array)[0][0])
    threshold = 0.5
    predicted_class = "healthy" if pred_prob >= threshold else "diseased"
    confidence = pred_prob if pred_prob >= threshold else 1 - pred_prob

    return jsonify({
        "class": predicted_class,
        "confidence": round(confidence, 4)  # return cleaner float
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
