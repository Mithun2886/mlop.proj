import math
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
REGISTERED_MODEL_NAME = "CaliforniaHousePriceModel"
MODEL_ALIAS = "champion"
FEATURES = [
	"MedInc",
	"HouseAge",
	"AveRooms",
	"AveBedrms",
	"Population",
	"AveOccup",
	"Latitude",
	"Longitude",
]
app = Flask(__name__)
model_bundle = None


def get_model_bundle():
	global model_bundle
	mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
	client = MlflowClient()
	try:
		champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
		if model_bundle is not None and model_bundle.get("registry_version") == champion.version:
			return model_bundle

		model = mlflow.sklearn.load_model(
			f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
		)
		model_bundle = {
			"model": model,
			"features": FEATURES,
			"registry_version": champion.version,
		}
	except MlflowException:
		if model_bundle is not None:
			return model_bundle
		if not MODEL_PATH.is_file():
			raise FileNotFoundError(
				"No registered champion or local model found. Run `python train.py` first."
			)
		model_bundle = joblib.load(MODEL_PATH)
	return model_bundle


@app.get("/")
def index():
	return send_from_directory(ROOT, "index.html")


@app.post("/predict")
def predict():
	payload = request.get_json(silent=True)
	if not isinstance(payload, dict):
		return jsonify(error="Send the property features as a JSON object."), 400

	try:
		bundle = get_model_bundle()
	except FileNotFoundError as error:
		return jsonify(error=str(error)), 503

	features = bundle["features"]
	missing = [feature for feature in features if feature not in payload]
	if missing:
		return jsonify(error=f"Missing property features: {', '.join(missing)}."), 400

	values = {}
	for feature in features:
		raw_value = payload[feature]
		if isinstance(raw_value, bool):
			return jsonify(error=f"{feature} must be a number."), 400
		try:
			value = float(raw_value)
		except (TypeError, ValueError):
			return jsonify(error=f"{feature} must be a number."), 400
		if not math.isfinite(value):
			return jsonify(error=f"{feature} must be a finite number."), 400
		values[feature] = value

	feature_frame = pd.DataFrame([values], columns=features)
	estimate = float(bundle["model"].predict(feature_frame)[0]) * 100_000
	return jsonify(estimate=round(estimate), currency="USD")


if __name__ == "__main__":
	app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))