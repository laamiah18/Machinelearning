from flask import Flask, render_template, request
import pickle
import datetime

app = Flask(__name__)

# Load all models
models = {
    "Apple": pickle.load(open("model_apple.pkl", "rb")),
    "Banana": pickle.load(open("model_banana.pkl", "rb")),
    "Grapes": pickle.load(open("model_grapes.pkl", "rb")),
    "Mango": pickle.load(open("model_mango.pkl", "rb")),
    "Watermelon": pickle.load(open("model_Water Melons.pkl", "rb"))
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    fruit = request.form["fruit"]
    month_name = request.form["month"]

    # Convert month → number
    month_num = datetime.datetime.strptime(month_name, "%B").month

    # Get the correct model
    model = models[fruit]

    # Predict
    pred = model.predict([[month_num]])[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted price of {fruit} in {month_name}: ₹{pred:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)


















