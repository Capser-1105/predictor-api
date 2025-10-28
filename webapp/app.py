from flask import Flask, render_template, request
import os
from src.image_processing import extract_dice_values
from prediction import predict_next
from train import add_new_data

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            image_path = os.path.join("data/raw", file.filename)
            file.save(image_path)

            dice_values = extract_dice_values(image_path)
            prediction = predict_next(dice_values)
            prediction["features"] = ",".join(map(str, dice_values + [0, 0, 0]))
            prediction["predicted"] = prediction["total"]
            result = prediction

    return render_template("index.html", result=result)

@app.route("/update", methods=["POST"])
def update():
    actual_total = int(request.form["actual_total"])
    features = [int(x) for x in request.form["features"].split(",")]
    predicted = float(request.form["predicted"])
    add_new_data(features, predicted, actual_total)
    return "✅ Mô hình đã được cập nhật và tái huấn luyện!"

if __name__ == "__main__":
    app.run(debug=True)