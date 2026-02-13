import os
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


# KServe endpoint
MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "http://localhost:8080/v1/models/model:predict")

THRESHOLD = 0.50

FEATURE_ORDER = [
    "Years at Company", "Performance Rating", "Number of Promotions",
    "Overtime", "Education Level", "Number of Dependents",
    "Job Level", "Company Size", "Company Tenure", "Remote Work",
    "Company Reputation", "OverallSatisfaction", "Opportunities",
    "AnnualIncome", "AgeGroup", "RoleStagnationRatio", "TenureGap",
    "EarlyCompanyTenureRisk", "LongTenureLowRoleRisk"
]

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print('incoming-data: ', data)

    try:
        instance = [data[f] for f in FEATURE_ORDER]
        payload = {"instances": [instance]}

        resp = requests.post(MODEL_ENDPOINT, json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        # model.predict() now returns [[p_stay, p_leave], ...]
        probs = result["predictions"][0]
        p_stay = float(probs[0])
        p_leave = float(probs[1])

        prediction = int(p_leave >= THRESHOLD)

        if p_leave < 0.30:
            risk = "Low"
        elif p_leave < 0.60:
            risk = "Medium"
        else:
            risk = "High"

        return jsonify({
            "prediction": prediction,
            "p_leave": round(p_leave, 4),
            "p_stay": round(p_stay, 4),
            "risk": risk,
            "threshold": THRESHOLD
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
