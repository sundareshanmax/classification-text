from flask import Flask, render_template, request
import joblib
import os
import traceback

app = Flask(__name__, template_folder="templates")

MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
except Exception as e:
    print("Error loading model or vectorizer:", e)
    traceback.print_exc()
    model = None
    vectorizer = None

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    user_text = ""
    error = ""

    if request.method == "POST":
        try:
            user_text = request.form.get("user_input", "")
            if not user_text:
                prediction = "Please enter text."
            else:
                if model is None or vectorizer is None:
                    prediction = "Model not loaded."
                else:
                    vect = vectorizer.transform([user_text])
                    pred = model.predict(vect)[0]
                    prediction = f"Prediction: {pred}"
        except Exception as e:
            error = str(e)
            traceback.print_exc()

    return render_template("index.html", prediction=prediction, user_text=user_text, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
