from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The trained model is missing. Run `python train.py` before starting the app."
        )

    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or not {"model", "features"} <= bundle.keys():
        raise ValueError(f"The model bundle at {MODEL_PATH} has an invalid format.")
    return bundle


def predict_estimate(values: dict[str, float], bundle: dict[str, object]) -> int:
    features = bundle["features"]
    feature_frame = pd.DataFrame([values], columns=features)
    prediction = bundle["model"].predict(feature_frame)[0]
    return round(float(prediction) * 100_000)


def main():
    st.set_page_config(
        page_title="California Home Value Estimator",
        page_icon="🏡",
        layout="wide",
    )
    st.title("California Home Value Estimator")
    st.write(
        "Explore a model estimate using characteristics of a California census block."
    )
    st.info(
        "This model uses 1990 California Housing data. Results are historical "
        "benchmarks, not current property valuations."
    )

    try:
        bundle = load_model_bundle()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    with st.form("prediction_form"):
        st.subheader("Property and neighborhood details")
        st.caption("Enter the census-block averages and location.")

        left, right = st.columns(2)
        with left:
            median_income = st.number_input(
                "Median income",
                min_value=0.5,
                max_value=15.0,
                value=8.3,
                step=0.1,
                help="Median household income, in units of $10,000.",
            )
            house_age = st.number_input(
                "Median house age",
                min_value=1,
                max_value=52,
                value=41,
                step=1,
                help="Median age of houses in the census block, in years.",
            )
            average_rooms = st.number_input(
                "Average rooms per household",
                min_value=0.5,
                max_value=142.0,
                value=7.0,
                step=0.1,
            )
            average_bedrooms = st.number_input(
                "Average bedrooms per household",
                min_value=0.3,
                max_value=35.0,
                value=1.0,
                step=0.1,
            )

        with right:
            population = st.number_input(
                "Neighborhood population",
                min_value=1,
                max_value=35682,
                value=322,
                step=1,
                help="Total population of the census block.",
            )
            average_occupancy = st.number_input(
                "Average household occupancy",
                min_value=0.5,
                max_value=1244.0,
                value=2.6,
                step=0.1,
                help="Average number of people per household.",
            )
            latitude = st.number_input(
                "Latitude",
                min_value=32.5,
                max_value=42.0,
                value=37.9,
                step=0.01,
            )
            longitude = st.number_input(
                "Longitude",
                min_value=-124.4,
                max_value=-114.3,
                value=-122.2,
                step=0.01,
            )

        submitted = st.form_submit_button(
            "Estimate home value", type="primary", use_container_width=True
        )

    if submitted:
        values = {
            "MedInc": median_income,
            "HouseAge": house_age,
            "AveRooms": average_rooms,
            "AveBedrms": average_bedrooms,
            "Population": population,
            "AveOccup": average_occupancy,
            "Latitude": latitude,
            "Longitude": longitude,
        }
        st.session_state["estimate"] = predict_estimate(values, bundle)

    estimate = st.session_state.get("estimate")
    if estimate is not None:
        st.divider()
        st.subheader("Estimated median home value")
        st.metric(label="Model estimate (USD)", value=f"${estimate:,.0f}")
        st.caption(
            "The dataset target is expressed in units of $100,000. "
            "This estimate reflects 1990 census data."
        )


if __name__ == "__main__":
    main()
