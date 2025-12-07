from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# ==== LOAD MODEL ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hipertensi_stacking.pkl")

model = joblib.load(MODEL_PATH)

# Urutan fitur HARUS sama dengan X_train.columns di Colab
FEATURE_ORDER = [
    "male",
    "age",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]


def to_float(value, default=0.0):
    if value is None or value == "":
        return default
    return float(value)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", sudah_prediksi=False)


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    # susun vektor fitur sesuai FEATURE_ORDER (12 fitur)
    x = [to_float(form.get(name)) for name in FEATURE_ORDER]
    features = np.array([x])

    # probabilitas kelas "risiko tinggi" (label 1)
    proba_high = float(model.predict_proba(features)[0, 1])
    prob_percent = round(proba_high * 100, 2)

    if proba_high < 0.33:
        kategori = "Risiko Rendah"
    elif proba_high < 0.66:
        kategori = "Risiko Sedang"
    else:
        kategori = "Risiko Tinggi"

    return render_template(
        "index.html",
        sudah_prediksi=True,
        prediction=kategori,
        prob=prob_percent,
        age=form.get("age"),
        sysBP=form.get("sysBP"),
        diaBP=form.get("diaBP"),
        BMI=form.get("BMI"),
        glucose=form.get("glucose"),
        heartRate=form.get("heartRate"),
    )


# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
