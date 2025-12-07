# HyperStack – Prediksi Risiko Hipertensi

Aplikasi web berbasis **Flask** yang memanfaatkan model **stacking ensemble** untuk
memprediksi risiko hipertensi berdasarkan data klinis pasien. Proyek ini dikembangkan
sebagai media edukasi, pendukung pengambilan keputusan awal, serta sarana presentasi
hasil penelitian machine learning di bidang kesehatan.

---

## 1. Gambaran Umum

**HyperStack** mengolah data klinis seperti usia, jenis kelamin, tekanan darah, BMI,
riwayat merokok, dan parameter lain menjadi sebuah **skor risiko hipertensi**.
Aplikasi akan menampilkan:

- python app.py
- source venv/Scripts/activate
- kategori risiko: **Rendah / Sedang / Tinggi**
- probabilitas risiko tinggi (dalam persen)
- ringkasan input yang dimasukkan pengguna
- penjelasan singkat yang berbeda untuk setiap tingkat risiko
- _medical disclaimer_ bahwa hasil tidak menggantikan diagnosis dokter

- Tampilan Website
![HyperStack Preview](material/HyperStack.png)

Model yang digunakan merupakan **stacking classifier** dengan base learner
XGBoost, LightGBM, dan CatBoost, serta meta-learner Logistic Regression.

---

## 2. Fitur Utama

- 🌐 **Antarmuka web responsif** dengan Bootstrap 5
- 🧮 **Prediksi risiko hipertensi** menggunakan model stacking ensemble
- 📊 **Tampilan hasil yang informatif**:
  - kategori risiko
  - probabilitas risiko tinggi
  - ringkasan data pasien
  - penjelasan rekomendasi singkat sesuai kategori
- 🧑‍⚕️ **Disclaimer medis** yang menegaskan bahwa aplikasi tidak menggantikan
  pemeriksaan dan keputusan dokter
- 🧪 **Tombol contoh kasus**:
  - _Contoh Pasien Risiko Rendah_
  - _Contoh Pasien Risiko Tinggi_
- 🧱 **Struktur proyek rapi**: pemisahan model, template HTML, dan aset statis (CSS)

---

## 3. Teknologi yang Digunakan

- **Backend**
  - Python 3.x
  - Flask
- **Machine Learning**
  - scikit-learn
  - XGBoost
  - LightGBM
  - CatBoost
  - joblib (untuk menyimpan & memuat model)
- **Frontend**
  - HTML5 + Jinja2 templating (Flask)
  - CSS (custom) + Bootstrap 5
  - JavaScript (sederhana, untuk tooltip & tombol preset)

---

## 4. Dataset & Pemodelan (Ringkas)

> Bagian ini menjelaskan secara singkat bagaimana model dibangun di notebook / Colab.

- Dataset klinis berisi sekitar **4.240** baris data pasien.
- Terdapat **13 fitur awal**, di antaranya:
  - jenis kelamin, usia
  - status merokok dan jumlah rokok per hari
  - penggunaan obat tekanan darah (BPMeds)
  - riwayat hipertensi, diabetes
  - kolesterol total, tekanan darah sistolik/diastolik
  - BMI, detak jantung, kadar glukosa
- Tahapan pengolahan data (di notebook):
  - pembersihan dan imputasi nilai hilang
  - normalisasi/standarisasi fitur numerik
  - penyeimbangan kelas menggunakan **SMOTE**
- Model akhir:
  - **Base learners**: XGBoost, LightGBM, CatBoost
  - **Meta learner**: Logistic Regression
  - **Skema**: `StackingClassifier` scikit-learn
- Metrik evaluasi (contoh hasil):
  - Akurasi ± **92,65 %**
  - Precision, recall, dan F1-score sekitar **0,92–0,93**
  - Analisis dilakukan menggunakan confusion matrix, ROC curve, dan PR curve.
- Model terlatih disimpan sebagai file:
  - `models/hipertensi_stacking.pkl`

---

## 5. Arsitektur Aplikasi

Alur kerja aplikasi:

1. Pengguna membuka halaman utama (`/`).
2. Pengguna mengisi form dengan data klinis (usia, jenis kelamin, tekanan darah, dll).
3. Form dikirim (`POST`) ke endpoint `/predict`.
4. Backend Flask:
   - mengambil nilai dari form
   - menyusun vektor fitur dalam urutan yang sama seperti saat training
   - memanggil `model.predict_proba()`
   - mengonversi probabilitas menjadi kategori risiko
5. Hasil prediksi dikirim kembali ke template `index.html` dan ditampilkan di panel hasil.

---

## 6. Struktur Folder Proyek

```text
prediksi_hipertensi/
├── app.py                    # Aplikasi Flask utama
├── models/
│   └── hipertensi_stacking.pkl   # Model stacking terlatih (joblib/pickle)
├── templates/
│   └── index.html            # Halaman utama (landing page + form + hasil)
└── static/
    └── css/
        └── style.css         # Style kustom untuk tampilan HyperStack
