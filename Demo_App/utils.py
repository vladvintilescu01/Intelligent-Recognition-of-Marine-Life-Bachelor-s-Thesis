import numpy as np
import tensorflow as tf
from PIL import Image
import os

CLASSES = np.array([
    "Catfish", "Glass Perchlet", "Goby", "Gourami",
    "Grass Carp", "Knifefish", "Silver Barb", "Tilapia"
])

def load_model_by_name(model_name: str, base_path="weights_and_structure"):
    h5_path = os.path.join(base_path, f"{model_name}.h5")
    if os.path.exists(h5_path):
        model = tf.keras.models.load_model(h5_path)
        return model
    else:
        raise FileNotFoundError(f"Model file not found: {h5_path}")

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Preprocess the uploaded image to match the model's expected input.
    """
    image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict_image(model, image: Image.Image):
    """
    Get the predicted class and confidence for a given image.
    """
    processed = preprocess_image(image)
    preds = model.predict(processed)
    pred_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    return CLASSES[pred_idx], confidence
