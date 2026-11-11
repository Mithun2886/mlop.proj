import unittest
from unittest.mock import patch

import app


FEATURES = app.FEATURES
VALID_PAYLOAD = {
    "MedInc": 8.3252,
    "HouseAge": 41,
    "AveRooms": 6.984127,
    "AveBedrms": 1.02381,
    "Population": 322,
    "AveOccup": 2.555556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


class StubModel:
    def __init__(self):
        self.received_features = None

    def predict(self, features):
        self.received_features = features
        return [2.345678]


class PredictApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()
        self.model = StubModel()
        self.model_bundle = {"model": self.model, "features": FEATURES}
        self.get_model_bundle = patch.object(
            app, "get_model_bundle", return_value=self.model_bundle
        )
        self.get_model_bundle.start()
        self.addCleanup(self.get_model_bundle.stop)

    def test_predict_returns_usd_estimate(self):
        response = self.client.post("/predict", json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(), {"estimate": 234568, "currency": "USD"}
        )
        self.assertEqual(list(self.model.received_features.columns), FEATURES)

    def test_predict_rejects_non_object_json(self):
        response = self.client.post("/predict", json=[1, 2, 3])

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_predict_reports_missing_features(self):
        payload = VALID_PAYLOAD.copy()
        del payload["MedInc"]

        response = self.client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("MedInc", response.get_json()["error"])

    def test_predict_rejects_non_numeric_and_non_finite_values(self):
        for value in ("not-a-number", True, float("nan"), float("inf")):
            with self.subTest(value=value):
                payload = VALID_PAYLOAD.copy()
                payload["MedInc"] = value

                response = self.client.post("/predict", json=payload)

                self.assertEqual(response.status_code, 400)
                self.assertIn("MedInc", response.get_json()["error"])

    def test_predict_reports_unavailable_model(self):
        with patch.object(
            app, "get_model_bundle", side_effect=FileNotFoundError("Model missing")
        ):
            response = self.client.post("/predict", json=VALID_PAYLOAD)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json(), {"error": "Model missing"})


if __name__ == "__main__":
    unittest.main()
