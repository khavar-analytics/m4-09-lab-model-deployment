from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.joblib")
target_names = joblib.load("target_names.joblib")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None or "features" not in data:
        return jsonify({"error": "Missing 'features' key in request body"}), 400

    features = data["features"]

    if len(features) != 4:
        return jsonify({
            "error": f"Expected exactly 4 feature values, got {len(features)}"
        }), 400

    for i, v in enumerate(features):
        if not isinstance(v, (int, float)):
            return jsonify({
                "error": f"All values must be numeric. Got non-numeric at index {i}: {v!r}"
            }), 400

    X = np.array(features).reshape(1, -1)
    pred_idx = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return jsonify({
        "predicted_class": str(target_names[pred_idx]),
        "probabilities": {
            str(target_names[i]): round(float(p), 4)
            for i, p in enumerate(proba)
        }
    }), 200


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json()

    if data is None or "samples" not in data:
        return jsonify({"error": "Missing 'samples' key in request body"}), 400

    samples = data["samples"]

    if not isinstance(samples, list) or len(samples) == 0:
        return jsonify({"error": "'samples' must be a non-empty list"}), 400

    for i, sample in enumerate(samples):
        if not isinstance(sample, list) or len(sample) != 4:
            return jsonify({
                "error": f"Sample at index {i} must be a list of exactly 4 values"
            }), 400
        for j, v in enumerate(sample):
            if not isinstance(v, (int, float)):
                return jsonify({
                    "error": f"Non-numeric value at sample {i}, index {j}: {v!r}"
                }), 400

    X = np.array(samples)
    pred_idxs = model.predict(X)
    probas = model.predict_proba(X)

    results = []
    for pred_idx, proba in zip(pred_idxs, probas):
        results.append({
            "predicted_class": str(target_names[pred_idx]),
            "probabilities": {
                str(target_names[i]): round(float(p), 4)
                for i, p in enumerate(proba)
            }
        })

    return jsonify({"predictions": results}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)