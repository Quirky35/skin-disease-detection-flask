import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

# Load model
model = tf.keras.models.load_model("../models/skin_disease_model.h5")

# Define class labels (must match your training)
class_names = [
    'actinic_keratoses',
    'basal_cell_carcinoma',
    'benign_keratosis',
    'dermatofibroma',
    'melanocytic_nevi',
    'melanoma',
    'vascular_lesions'
]

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ File not found: {img_path}")
        return

    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Predict
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]
    confidence = predictions[0][class_index] * 100

    print(f"✅ Prediction: {class_name} ({confidence:.2f}%)")

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])
