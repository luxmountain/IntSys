# ---Phan Thị Hồng Thắm - B22DCCN806---

# App.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle

app = Flask(__name__)

# Load model và scaler
model = keras.models.load_model('diabetes_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        data = request.get_json()

        pregnancies = float(data['pregnancies'])
        glucose = float(data['glucose'])
        blood_pressure = float(data['blood_pressure'])
        skin_thickness = float(data['skin_thickness'])
        insulin = float(data['insulin'])
        bmi = float(data['bmi'])
        diabetes_pedigree = float(data['diabetes_pedigree'])
        age = float(data['age'])

        # Chuẩn bị dữ liệu
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, diabetes_pedigree, age]])

        # Chuẩn hóa
        input_data_scaled = scaler.transform(input_data)

        # Dự đoán
        prediction_proba = model.predict(input_data_scaled, verbose=0)
        probability = float(prediction_proba[0][0])
        prediction = int(prediction_proba[0][0] > 0.5)

        result = {
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'status': 'success'
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)

# 74