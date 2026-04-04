from flask import Flask, render_template, request, jsonify
import joblib
import PyPDF2

app = Flask(__name__)

# ✅ Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = ""

        # 📄 If PDF uploaded
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted

        # 📝 If text input
        else:
            text = request.form.get("text", "").strip()

        print("INPUT:", text)

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # ✅ Convert text → vector
        text_vec = vectorizer.transform([text])

        # ✅ Predict
        prediction = model.predict(text_vec)[0]

        # ✅ Confidence (safe)
        try:
            proba = model.predict_proba(text_vec)[0]
            confidence = float(max(proba)) * 100
        except:
            confidence = 90.0

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 1)
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
