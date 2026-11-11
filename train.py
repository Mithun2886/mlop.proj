from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
RANDOM_STATE = 42
CONFIGURATIONS = [
    {
        "run_name": "standard-scaler-baseline",
        "scaler": "StandardScaler",
        "test_size": 0.2,
        "n_estimators": 120,
        "max_depth": 18,
        "min_samples_leaf": 2,
    },
    {
        "run_name": "minmax-scaler-baseline",
        "scaler": "MinMaxScaler",
        "test_size": 0.2,
        "n_estimators": 120,
        "max_depth": 18,
        "min_samples_leaf": 2,
    },
    {
        "run_name": "no-scaling-baseline",
        "scaler": "None",
        "test_size": 0.2,
        "n_estimators": 120,
        "max_depth": 18,
        "min_samples_leaf": 2,
    },
    {
        "run_name": "standard-scaler-tuned-forest",
        "scaler": "StandardScaler",
        "test_size": 0.2,
        "n_estimators": 200,
        "max_depth": 24,
        "min_samples_leaf": 1,
    },
]


def build_model(configuration: dict[str, object]) -> Pipeline:
    scaler_name = configuration["scaler"]
    if scaler_name == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_name == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = "passthrough"

    return Pipeline(
        steps=[
            ("scaler", scaler),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=int(configuration["n_estimators"]),
                    max_depth=int(configuration["max_depth"]),
                    min_samples_leaf=int(configuration["min_samples_leaf"]),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    tracking_uri = f"sqlite:///{ROOT / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("California Housing Price Prediction")

    housing = fetch_california_housing(as_frame=True)
    features = housing.data[FEATURES]
    target = housing.target
    results = []

    for configuration in CONFIGURATIONS:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=float(configuration["test_size"]),
            random_state=RANDOM_STATE,
        )
        model = build_model(configuration)

        with mlflow.start_run(run_name=str(configuration["run_name"])):
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = {
                "rmse": mean_squared_error(y_test, predictions) ** 0.5,
                "mae": mean_absolute_error(y_test, predictions),
                "r2": r2_score(y_test, predictions),
            }

            mlflow.log_params(
                {
                    **configuration,
                    "model_type": "RandomForestRegressor",
                    "random_state": RANDOM_STATE,
                    "dataset": "California Housing",
                }
            )
            mlflow.log_metrics(metrics)
            logged_model = mlflow.sklearn.log_model(
                sk_model=model,
                name="house-price-model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
                signature=infer_signature(X_test, predictions),
                input_example=X_test.head(3),
            )
            results.append(
                {
                    "run_id": mlflow.active_run().info.run_id,
                    "model": model,
                    "metrics": metrics,
                    "run_name": configuration["run_name"],
                    "model_uri": logged_model.model_uri,
                }
            )

    best_result = max(results, key=lambda result: result["metrics"]["r2"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_result["model"], "features": FEATURES}, MODEL_PATH)

    client = MlflowClient(tracking_uri=tracking_uri)
    for result in results:
        selection = "best_by_r2" if result is best_result else "compared"
        client.set_tag(result["run_id"], "model_selection", selection)
    client.log_artifact(best_result["run_id"], str(MODEL_PATH), artifact_path="deployment")

    try:
        champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS)
        champion_r2 = float(champion.tags.get("r2", "-inf"))
    except MlflowException:
        champion = None
        champion_r2 = float("-inf")

    registered_version = mlflow.register_model(
        model_uri=best_result["model_uri"],
        name=REGISTERED_MODEL_NAME,
        await_registration_for=300,
    )
    for key, value in {
        "mae": best_result["metrics"]["mae"],
        "rmse": best_result["metrics"]["rmse"],
        "r2": best_result["metrics"]["r2"],
        "source_run": best_result["run_id"],
        "selection_metric": "r2",
    }.items():
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME, str(registered_version.version), key, str(value)
        )

    promoted = best_result["metrics"]["r2"] > champion_r2
    if promoted:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, MODEL_ALIAS, str(registered_version.version)
        )

    for result in sorted(results, key=lambda item: item["metrics"]["r2"], reverse=True):
        metrics = result["metrics"]
        print(
            f"{result['run_name']}: MAE={metrics['mae']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}"
        )
    print(f"Best run by R2: {best_result['run_name']}")
    print(
        f"Registered {REGISTERED_MODEL_NAME} version {registered_version.version}; "
        f"{'promoted to' if promoted else 'kept existing'} {MODEL_ALIAS} alias."
    )
    print(f"Model saved to {MODEL_PATH}")
    print(f"MLflow tracking database: {ROOT / 'mlflow.db'}")


if __name__ == "__main__":
    main()