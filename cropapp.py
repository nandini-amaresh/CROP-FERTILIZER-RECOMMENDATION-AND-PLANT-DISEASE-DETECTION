from flask import Flask, render_template, request, url_for
import numpy as np
import joblib
import pickle
import os
from PIL import Image
import tensorflow as tf
import json
from werkzeug.utils import secure_filename



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')




# Load crop recommendation models
model = joblib.load('model.pkl')
scaler = joblib.load('minmaxscaler.pkl')

# Load fertilizer recommendation models
with open('fertilizer_model.pkl', 'rb') as f:
    fert_model = pickle.load(f)

with open('fertilizer_scaler.pkl', 'rb') as f:
    fert_scaler = pickle.load(f)

with open('le_soil.pkl', 'rb') as f:
    soil_encoder = pickle.load(f)

with open('le_crop.pkl', 'rb') as f:
    crop_encoder = pickle.load(f)

with open('le_fert.pkl', 'rb') as f:
    fertilizer_encoder = pickle.load(f)

# Load TensorFlow model for plant disease detection
disease_model = tf.keras.models.load_model('plant_disease_model.h5')

# Load class indices mapping
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
    # Reverse the dict to get idx -> class_name
    idx_to_class = {v: k for k, v in class_indices.items()}

img_size = 128  # Same size used during training

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

def match_class(value, encoder_classes):
    """Match input with encoder classes (case-insensitive)."""
    value_clean = value.strip().lower()
    for cls in encoder_classes:
        if cls.lower() == value_clean:
            return cls
    return None

def recommend_fertilizer(temp, humidity, moisture, soil, crop, N, P, K):
    matched_soil = match_class(soil, soil_encoder.classes_)
    matched_crop = match_class(crop, crop_encoder.classes_)

    if matched_soil is None:
        return f"Invalid soil type: '{soil}'. Valid types are: {list(soil_encoder.classes_)}"
    if matched_crop is None:
        return f"Invalid crop type: '{crop}'. Valid types are: {list(crop_encoder.classes_)}"

    soil_code = soil_encoder.transform([matched_soil])[0]
    crop_code = crop_encoder.transform([matched_crop])[0]

    features = np.array([[temp, humidity, moisture, soil_code, crop_code, N, K, P]])
    scaled = fert_scaler.transform(features)
    prediction = fert_model.predict(scaled)
    return fertilizer_encoder.inverse_transform(prediction)[0]

def predict_disease_from_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0  # Normalize as in training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = disease_model.predict(img_array)
    predicted_idx = np.argmax(predictions, axis=1)[0]
    predicted_class = idx_to_class[predicted_idx]
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def index():
    crop = None
    fertilizer = None
    disease_result = None
    uploaded_image_url  = None

    if request.method == 'POST':
        if 'predict_crop' in request.form:
            try:
                N = float(request.form['N'])
                P = float(request.form['P'])
                K = float(request.form['K'])
                temperature = float(request.form['temperature'])
                humidity = float(request.form['humidity'])
                ph = float(request.form['ph'])
                rainfall = float(request.form['rainfall'])

                features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                crop = crop_dict.get(prediction, "Unknown")
            except Exception as e:
                crop = f"Error: {e}"

        elif 'predict_fertilizer' in request.form:
            try:
                temp = float(request.form['fert_temperature'])
                humidity = float(request.form['fert_humidity'])
                moisture = float(request.form['moisture'])
                soil_type = request.form['soil']
                crop_type = request.form['fert_crop']
                N = int(request.form['fert_nitrogen'])
                P = int(request.form['fert_phosphorous'])
                K = int(request.form['fert_potassium'])

                fertilizer = recommend_fertilizer(temp, humidity, moisture, soil_type, crop_type, N, P, K)
            except Exception as e:
                fertilizer = f"Error: {e}"

        elif 'predict_disease' in request.form:
            try:
                if 'plant_image' not in request.files:
                    disease_result = "No image uploaded."
                else:
                    file = request.files['plant_image']
                    if file.filename == '':
                        disease_result = "No selected file."
                    else:
                        filename = secure_filename(file.filename)
                        filename = secure_filename(file.filename)
                        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(filepath)
                        disease_result = predict_disease_from_image(filepath)
                        uploaded_image_url = url_for('static', filename=f'uploads/{filename}')


            except Exception as e:
                disease_result = f"Error: {e}"

    return render_template(
    'index.html',
    crop=crop,
    fertilizer=fertilizer,
    disease_result=disease_result,
    uploaded_image_url=uploaded_image_url  
)
from flask import send_from_directory


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
