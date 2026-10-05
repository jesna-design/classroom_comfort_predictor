import os
import csv
from datetime import datetime
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from train_model import train, FEATURES, MODEL_FILE

app = Flask(__name__)
LOG_FILE = "predictions_log.csv"

if not os.path.exists(MODEL_FILE):
    train()
bundle = joblib.load(MODEL_FILE)
model = bundle["model"]

# field: (min, max) - used to validate form input before predicting
LIMITS = {
    "length": (3, 25),
    "width": (3, 20),
    "height": (2, 6),
    "students": (0, 200),
    "airflow": (0, 200),
    "heat": (0, 30000),
    "lighting": (0, 2000),
    "noise": (20, 120),
    "layout": (0, 2),
    "visual_access": (0, 100),
    "greenery": (0, 100),
    "time_of_day": (0, 24),
}

LAYOUT_NAMES = {0: "Rows", 1: "Clusters", 2: "Circles"}


def get_suggestions(d, label):
    if label == "Comfortable":
        return ["Conditions look good. Keep the current room setup as is."]
    tips = []
    per_student_area = (d["length"] * d["width"]) / max(d["students"], 1)
    if per_student_area < 0.9:
        tips.append("Room feels crowded for this many students: reduce occupancy or move to a bigger room.")
    if d["airflow"] / max(d["students"], 1) < 0.5:
        tips.append("Airflow is low for this many students: improve ventilation or use fans/AC.")
    if d["lighting"] < 400:
        tips.append("Lighting is low: add more light sources near work areas.")
    elif d["lighting"] > 900:
        tips.append("Lighting may be too harsh: consider dimming or diffusing it.")
    if d["noise"] > 70:
        tips.append("Noise level is high: reduce crowd noise or improve sound insulation.")
    if d["visual_access"] < 55:
        tips.append("Visual accessibility is low: rearrange seating so all students can see the board/screen.")
    if d["greenery"] < 2:
        tips.append("Very little greenery: a few plants can modestly improve comfort.")
    if d["heat"] / max(d["students"], 1) > 110:
        tips.append("Heat generation per student is high: improve cooling or reduce equipment load.")
    return tips or ["Try adjusting layout, airflow or lighting and check again."]


@app.route("/")
def index():
    return render_template("index.html", model_name=bundle["name"],
                           accuracy=round(bundle["accuracy"] * 100, 1))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    values = {}
    for f in FEATURES:
        try:
            v = float(data.get(f))
        except (TypeError, ValueError):
            return jsonify(error=f"Please enter a valid value for {f.replace('_', ' ')}."), 400
        lo, hi = LIMITS[f]
        if not lo <= v <= hi:
            return jsonify(error=f"{f.replace('_', ' ').title()} must be between {lo} and {hi}."), 400
        values[f] = v

    X = pd.DataFrame([values])[FEATURES]
    label = model.predict(X)[0]
    probs = {c: round(float(p) * 100, 1)
             for c, p in zip(model.classes_, model.predict_proba(X)[0])}

    room = str(data.get("room", "")).strip()[:40] or "Unnamed room"
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["time", "room"] + FEATURES + ["result"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), room]
                   + [values[f] for f in FEATURES] + [label])

    return jsonify(result=label, probabilities=probs,
                   suggestions=get_suggestions(values, label))


@app.route("/history")
def history():
    if not os.path.exists(LOG_FILE):
        return jsonify([])
    df = pd.read_csv(LOG_FILE).tail(8).iloc[::-1]
    return jsonify(df[["time", "room", "students", "result"]].to_dict("records"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
