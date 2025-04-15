from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from io import BytesIO  # <-- Import this to fix image stream issue

app = Flask(__name__)

MODEL_DIR = "model"
MODEL_PATH = os.getenv("MODEL_PATH", f"{MODEL_DIR}/vgg16_rgb_final_model.h5")

model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)

CLASS_NAMES = ['diseased', 'healthy']

def prepare_image(img_file):
    # Convert to BytesIO for keras compatibility
    img = image.load_img(BytesIO(img_file.read()), target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
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

    prediction = model.predict(img_array)
    class_idx = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({
        "class": CLASS_NAMES[class_idx],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
