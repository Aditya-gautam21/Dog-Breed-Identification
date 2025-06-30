import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

def predict(img_path, model_path='dog_breed_model.h5', target_size=(224, 224), class_indices=None):
    model = load_model(model_path)

    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    class_idx = np.argmax(prediction)
    
    if class_indices:
        idx_to_class = {v: k for k, v in class_indices.items()}
        print(f"Prediction: {idx_to_class[class_idx]}")
    else:
        print(f"Prediction class index: {class_idx}")
