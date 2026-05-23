from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load models
logistic = pickle.load(open("models/logistic.pkl", "rb"))
knn = pickle.load(open("models/knn.pkl", "rb"))
svm = pickle.load(open("models/svm.pkl", "rb"))

flower_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sl = float(request.form["sepal_length"])
    sw = float(request.form["sepal_width"])
    pl = float(request.form["petal_length"])
    pw = float(request.form["petal_width"])
    algo = request.form["algorithm"]

    features = np.array([[sl, sw, pl, pw]])

    if algo == "Logistic Regression":
        pred = logistic.predict(features)
    elif algo == "KNN":
        pred = knn.predict(features)
    else:
        pred = svm.predict(features)

    result = flower_names[pred[0]]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Flower: {result}",
        algorithm_text=f"Algorithm Used: {algo}"
    )

if __name__ == "__main__":
    app.run(debug=True)
