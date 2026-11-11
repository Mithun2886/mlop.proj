# California Home Value Estimator

A Streamlit application and Flask API that predict median home values from California census-block features using a random forest trained on the scikit-learn California Housing dataset. The dataset target is expressed in units of $100,000 and comes from 1990 census data; estimates are historical benchmarks, not current appraisals.

## Run locally

Install the dependencies and train the model. The first training run downloads the public dataset and records four runs in the local MLflow SQLite database. The runs compare `StandardScaler`, `MinMaxScaler`, no scaling, and a more complex random forest. All use the same 80/20 split and random seed so the metrics are comparable. The best R² run is registered as a version of `CaliforniaHousePriceModel`; its MAE, RMSE, R², and source run are saved as version tags. The registry's `champion` alias moves only when a new candidate beats the existing champion's R². The app loads from this alias, with the local artifact as a fallback.

```powershell
python -m pip install -r requirements.txt
python train.py
python -m streamlit run streamlit_app.py
```

Open the Streamlit interface at `http://localhost:8501`. To run the Flask API as well, start `python app.py` in a second terminal; it listens at `http://127.0.0.1:5000`.

## Docker deployment

Build the image and start both the Streamlit interface and Flask API:

```powershell
docker build -t california-home-estimator .
docker run --name california-home-estimator --rm -p 8501:8501 -p 5000:5000 california-home-estimator
```

Open the interface at `http://localhost:8501`; the prediction API remains available at `http://localhost:5000/predict`. Docker trains and packages the model during the image build, so the running container does not download the dataset or need a mounted model file. The first image build requires internet access to download the California Housing dataset and can take several minutes.

## Prediction API

Send a JSON object with all eight model features to `POST http://127.0.0.1:5000/predict`. The response estimate is in USD; the underlying California Housing dataset is historical (1990) and its target is in units of $100,000.

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"MedInc":8.3252,"HouseAge":41,"AveRooms":6.984127,"AveBedrms":1.02381,"Population":322,"AveOccup":2.555556,"Latitude":37.88,"Longitude":-122.23}'
```

Example response:

```json
{
  "estimate": 423020,
  "currency": "USD"
}
```

In Postman, choose **POST**, enter `http://127.0.0.1:5000/predict`, then select **Body > raw > JSON** and send the same feature object. Missing or invalid feature values return HTTP 400; if no trained model is available, the API returns HTTP 503. Additional JSON keys are ignored.

## Explore MLflow

In a second terminal from the project directory, start the MLflow tracking UI:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Open `http://127.0.0.1:5001`, select the **California Housing Price Prediction** experiment, and open **Training runs**. Compare each run's `scaler`, `test_size`, `n_estimators`, `max_depth`, and `min_samples_leaf` parameters alongside the `mae`, `rmse`, and `r2` metrics. The best run is tagged `model_selection=best_by_r2`. To inspect registered versions and the `champion` alias, open the **Models** section in MLflow. Run `python train.py` again to add another set of four runs and register its best candidate.

## Model details

The model uses median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, and longitude. Preprocessing is inside each sklearn pipeline, so scalers are fitted only on the training split. Random forests are generally insensitive to feature scaling; the scaler runs demonstrate that while the tuned run tests estimator hyperparameters. Training logs its parameters and holdout metrics in MLflow and saves the best serving pipeline under `models/`.