# Mengimpor library bawaan Python untuk membaca dan mem-parsing file berformat JSON
import json

# Mengimpor library pandas untuk menyusun data input menjadi tabel (DataFrame) sesuai format model
import pandas as pd

# Mengimpor Flask (framework web), request (menangkap data masuk), dan jsonify (mengubah output ke JSON)
from flask import Flask, request, jsonify

# Mengimpor joblib untuk memuat file model machine learning dan scaler yang sudah disimpan (.sav)
import joblib

# Mengimpor fungsi untuk membuat tampilan dokumentasi interaktif Swagger UI di Flask
from flask_swagger_ui import get_swaggerui_blueprint

# Membuat instance aplikasi web Flask
app = Flask(__name__)

# Mendaftarkan halaman Swagger UI pada alamat "/docs" yang mengambil spesifikasi API dari "/openapi.json"
app.register_blueprint(get_swaggerui_blueprint("/docs", "/openapi.json"), url_prefix="/docs")

# Membuat endpoint GET "/openapi.json" untuk menyediakan file spesifikasi OpenAPI bagi Swagger UI
@app.route("/openapi.json")
def openapi():
    # Membuka file "openapi.json" dari folder yang sama
    with open("openapi.json") as f:
        # Membaca isi file JSON dan mengirimkannya sebagai response HTTP JSON
        return jsonify(json.load(f))

# Daftar nama dan urutan 11 fitur yang wajib sesuai dengan urutan saat model dilatih (training)
FEATURE_NAMES = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

# Memuat model Decision Tree yang sudah dilatih dari folder "model"
dt = joblib.load("model/heartf_dt.sav")

# Memuat scaler (MinMaxScaler) untuk menyamakan skala angka input dengan data saat training
sc = joblib.load("model/scaler_heartf.sav")

# Membuat endpoint "/predict" yang hanya menerima metode HTTP POST (pengiriman data)
@app.route("/predict", methods=["POST"])
def predict():
    # Mengambil payload JSON yang dikirim oleh user/client dari body request
    data = request.get_json()
    
    # Jika input berupa dictionary (key-value), ambil nilainya sesuai urutan FEATURE_NAMES; jika sudah list, langsung pakai
    row = [data[col] for col in FEATURE_NAMES] if isinstance(data, dict) else data
    
    # Mengubah list 1 dimensi menjadi DataFrame 2 dimensi (1 baris, 11 kolom) lengkap dengan nama kolomnya
    df = pd.DataFrame([row], columns=FEATURE_NAMES)

    X_scaled = sc.transform(df)
    
    # Melakukan normalisasi skala data (sc.transform), memprediksi dengan model (dt.predict), lalu mengubah hasil pertama ke integer Python
    pred = int(dt.predict(X_scaled)[0])
    # Ambil probabilitas kelas yang terpilih (indeks 0 untuk negative, indeks 1 untuk positive)
    prob = float(dt.predict_proba(X_scaled)[0][pred])
    
    # Mengembalikan hasil prediksi dalam format JSON beserta label teks ("positive" jika 1, "negative" jika 0)
    return jsonify({
        "prediction": pred,
        "label": "positive" if pred == 1 else "negative",
        "confidence": f"{prob * 100}%"
    })

# Memastikan server hanya berjalan jika file ini dieksekusi langsung (bukan di-import sebagai modul)
if __name__ == "__main__":
    # Menjalankan server lokal Flask dengan mode debug aktif (otomatis reload jika kode diubah)
    app.run(debug=True)