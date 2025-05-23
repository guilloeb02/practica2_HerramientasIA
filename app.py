from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar modelo y vectorizador
model = joblib.load("./model/model.pkl")
tfidf = joblib.load("./model/tfidf.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({"error": "No se proporcionó texto"}), 400

    vectorized_text = tfidf.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return jsonify({"prediction": str(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
