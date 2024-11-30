import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import Flask-CORS

# Membuat aplikasi Flask
app = Flask(__name__)

# Aktifkan CORS
CORS(app)  # Mengizinkan semua domain

# Muat model yang sudah disimpan
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Endpoint untuk memprediksi diabetes
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        # Ambil data dari request
        data = request.get_json()

        # Buat DataFrame dengan nama kolom yang sesuai
        input_data = pd.DataFrame([{
            "Pregnancies": data['Pregnancies'],
            "Glucose": data['Glucose'],
            "BloodPressure": data['BloodPressure'],
            "SkinThickness": data['SkinThickness'],
            "Insulin": data['Insulin'],
            "BMI": data['BMI'],
            "DiabetesPedigreeFunction": data['DiabetesPedigreeFunction'],
            "Age": data['Age']
        }])

        # Melakukan prediksi dengan model yang sudah dimuat
        prediction = model.predict(input_data)

        # Mendapatkan probabilitas prediksi
        probabilities = model.predict_proba(input_data)

        # Probabilitas positif dan negatif dalam bentuk persentase
        probability_negative = probabilities[0][0] * 100  # Probabilitas untuk kelas 0 (negatif)
        probability_positive = probabilities[0][1] * 100  # Probabilitas untuk kelas 1 (positif)

        # Prediksi output (0 atau 1, di mana 1 berarti positif diabetes)
        if prediction[0] == 1:
            result = f'Anda memiliki peluang menderita diabetes berdasarkan model KNN kami. Kemungkinan menderita diabetes adalah {probability_positive:.2f}%.'
        else:
            result = 'Hasil prediksi menunjukkan Anda kemungkinan rendah terkena diabetes.'

        # Kembalikan hasil prediksi dan probabilitas dalam bentuk JSON
        data = jsonify({
            'prediction': result,
            'probabilities': {
                'negative': f"{probability_negative:.2f}%",  # Format 2 desimal
                'positive': f"{probability_positive:.2f}%"
            }
        })

        return data

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/')
def index():
    return render_template('index.html')

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)



