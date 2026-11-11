import unittest

import pandas as pd

from streamlit_app import predict_estimate


class StubModel:
    def __init__(self):
        self.received_features = None

    def predict(self, features):
        self.received_features = features
        return [4.23019775]


class StreamlitPredictionTests(unittest.TestCase):
    def test_predict_estimate_uses_bundle_feature_order_and_usd_units(self):
        features = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
        values = dict.fromkeys(features, 1.0)
        model = StubModel()

        estimate = predict_estimate(
            values, {"model": model, "features": features}
        )

        self.assertEqual(estimate, 423020)
        self.assertEqual(list(model.received_features.columns), features)
        self.assertIsInstance(model.received_features, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
