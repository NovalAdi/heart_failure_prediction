# Heart Failure Prediction API

Backend Flask sederhana untuk memprediksi penyakit jantung dari 11 fitur klinis
menggunakan `DecisionTreeClassifier` scikit-learn yang sudah dilatih.

## Jalankan

    pip install -r requirements.txt
    python app.py

- API: `http://127.0.0.1:5000`
- Swagger UI: `http://127.0.0.1:5000/docs`

## Model

Isi folder `model/`:

| File | Isi |
|---|---|
| `heartf_dt.sav` | `DecisionTreeClassifier` (121 node, depth 10, 61 leaf) |
| `scaler_heartf.sav` | `MinMaxScaler` — sumber urutan & rentang fitur (`feature_names_in_`, `data_min_`, `data_max_`) |
| `encoder_heartf.sav` | `OrdinalEncoder` — **tersimpan tanpa fit** (`categories_` kosong), tidak dipakai |

Dataset latih: Heart Failure Prediction Dataset (918 baris, 11 fitur + target
`HeartDisease`). Akurasi model pada dataset tersebut: **0.8736**.

`encoder_heartf.sav` tidak bisa dipakai karena di-save sebelum `fit`. Pemetaan
kategorinya direkonstruksi di `app.py` mengikuti perilaku `OrdinalEncoder`
default saat training, yaitu **urut alfabetis** — inilah yang membuat akurasi
0.87, bukan pemetaan urutan medis (yang hanya menghasilkan 0.70).

## Urutan fitur

Diambil otomatis dari `scaler.feature_names_in_` saat aplikasi start:

| # | Fitur | Tipe | Rentang latih |
|---|---|---|---|
| 0 | Age | int | 28–77 |
| 1 | Sex | kategori | F, M |
| 2 | ChestPainType | kategori | ASY, ATA, NAP, TA |
| 3 | RestingBP | int | 0–200 |
| 4 | Cholesterol | int | 0–603 |
| 5 | FastingBS | int | 0–1 |
| 6 | RestingECG | kategori | LVH, Normal, ST |
| 7 | MaxHR | int | 63–202 |
| 8 | ExerciseAngina | kategori | N, Y |
| 9 | Oldpeak | float | -2.6–6.2 |
| 10 | ST_Slope | kategori | Down, Flat, Up |

## POST /predict

Kirim JSON object (key = nama fitur) atau array 11 nilai sesuai urutan di atas.
Fitur kategorikal boleh berupa string (lebih aman) atau kode integer.

    curl -X POST http://127.0.0.1:5000/predict \
      -H "Content-Type: application/json" \
      -d '{"Age":40,"Sex":"M","ChestPainType":"ATA","RestingBP":140,
           "Cholesterol":289,"FastingBS":0,"RestingECG":"Normal","MaxHR":172,
           "ExerciseAngina":"N","Oldpeak":0,"ST_Slope":"Up"}'

Response:

    {
      "prediction": 0,
      "label": "negative",
      "probability": 0.0
    }

- `prediction` — 0 = tidak terindikasi, 1 = terindikasi
- `probability` — probabilitas kelas positif (0–1)
- Error (400) — `{"error": "..."}` untuk fitur kurang/berlebih, kategori tidak
  dikenal, atau nilai non-numerik

## GET /health

    {"status": "ok", "features": [...]}

## Pemetaan kategori (untuk referensi)

    Sex            F=0, M=1
    ChestPainType  ASY=0, ATA=1, NAP=2, TA=3
    RestingECG     LVH=0, Normal=1, ST=2
    ExerciseAngina N=0, Y=1
    ST_Slope       Down=0, Flat=1, Up=2

## Catatan validasi

- Fitur kontinu di-*clamp* ke rentang latih, bukan ditolak. Scaler di-fit pada
  split train yang rentangnya lebih sempit dari dataset utuh (mis. `MaxHR` min
  63 vs 60 di data asli), jadi menolak nilai di luar rentang akan memblokir
  pasien valid.
- Fitur kategorikal divalidasi ketat: nilai di luar daftar kategori → 400.

contoh untuk hasil positif
{
  "Age": 58,
  "Sex": 1,
  "ChestPainType": 0,
  "RestingBP": 145,
  "Cholesterol": 265,
  "FastingBS": 0,
  "RestingECG": 1,
  "MaxHR": 130,
  "ExerciseAngina": 1,
  "Oldpeak": 1.8,
  "ST_Slope": 1
}