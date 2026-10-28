
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            variance = float(request.form["variance"])
            skewness = float(request.form["skewness"])
            curtosis = float(request.form["curtosis"])
            entropy = float(request.form["entropy"])

            features = np.array([[
                variance, skewness, curtosis, entropy
            ]])

            scaled_features = scaler.transform(features)
            prediction = model.predict(scaled_features)[0]

            if prediction == 0:
                result = "Genuine Banknote"
            else:
                result = "Counterfeit Banknote"

        except (ValueError, KeyError):
            error = "Please enter valid numerical values."

    return render_template(
        "index.html",
        result=result,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
