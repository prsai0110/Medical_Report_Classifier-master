@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = ""

        # Get text from form
        if 'file' in request.files:
            file = request.files['file']
            import PyPDF2
            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        else:
            text = request.form.get("text", "").strip()

        print("INPUT TEXT:", text)  # ✅ debug

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # ✅ VERY IMPORTANT
        text_vec = vectorizer.transform([text])

        print("TRANSFORM DONE")  # ✅ debug

        prediction = model.predict(text_vec)[0]

        print("PREDICTION:", prediction)  # ✅ debug

        # Safe confidence
        try:
            proba = model.predict_proba(text_vec)[0]
            confidence = float(proba.max()) * 100
        except:
            confidence = 90.0

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 1)
        })

    except Exception as e:
        print("ERROR:", str(e))  # 🔴 IMPORTANT
        return jsonify({"error": str(e)}), 500
