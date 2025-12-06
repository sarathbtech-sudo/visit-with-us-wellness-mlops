"""Streamlit inference app for Visit with Us - Wellness Tourism Package prediction."""
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path("tourism_project") / "model_building" / "model.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"Model file not found at {MODEL_PATH}. "
            "Please run the training pipeline and ensure the model is saved."
        )
        return None
    model = joblib.load(MODEL_PATH)
    return model


def main():
    st.title("Visit with Us - Wellness Tourism Purchase Prediction")
    st.write(
        "Predict whether a customer is likely to purchase the **Wellness Tourism Package** "
        "before contacting them."
    )

    model = load_model()

    with st.form("customer_form"):
        st.subheader("Customer Details")

        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Self Employed", "Freelancer", "Student", "Other"],
        )
        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP", "Other"],
        )

        city_tier = st.selectbox("City Tier", [1, 2, 3])
        monthly_income = st.number_input(
            "Monthly Income", min_value=0, max_value=10_000_000, value=50_000, step=1_000
        )
        num_trips = st.number_input(
            "Number of Trips (per year)", min_value=0, max_value=50, value=2, step=1
        )

        owns_car = st.selectbox("Own Car", ["No", "Yes"])
        has_passport = st.selectbox("Has Passport", ["No", "Yes"])

        num_persons_visiting = st.number_input(
            "Number of Persons Visiting", min_value=1, max_value=10, value=2, step=1
        )
        num_children_visiting = st.number_input(
            "Number of Children Visiting (under 5)",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
        )
        preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

        st.subheader("Interaction Details")
        typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
        product_pitched = st.selectbox(
            "Product Pitched",
            [
                "Basic",
                "Standard",
                "Deluxe",
                "Super Deluxe",
                "King",
                "Other",
            ],
        )
        pitch_satisfaction_score = st.slider(
            "Pitch Satisfaction Score", min_value=1, max_value=5, value=3
        )
        num_followups = st.number_input(
            "Number of Follow-ups", min_value=0, max_value=20, value=2, step=1
        )
        duration_of_pitch = st.number_input(
            "Duration of Pitch (minutes)", min_value=0, max_value=240, value=30, step=1
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        if model is None:
            st.error("Model not available. Please contact the administrator.")
            return

        input_dict = {
            "Age": age,
            "Gender": gender,
            "MaritalStatus": marital_status,
            "Occupation": occupation,
            "Designation": designation,
            "CityTier": city_tier,
            "MonthlyIncome": monthly_income,
            "NumberOfTrips": num_trips,
            "OwnCar": 1 if owns_car == "Yes" else 0,
            "Passport": 1 if has_passport == "Yes" else 0,
            "NumberOfPersonVisiting": num_persons_visiting,
            "NumberOfChildrenVisiting": num_children_visiting,
            "PreferredPropertyStar": preferred_property_star,
            "TypeofContact": typeof_contact,
            "ProductPitched": product_pitched,
            "PitchSatisfactionScore": pitch_satisfaction_score,
            "NumberOfFollowups": num_followups,
            "DurationOfPitch": duration_of_pitch,
        }

        df_input = pd.DataFrame([input_dict])
        proba = model.predict_proba(df_input)[:, 1][0]
        pred = int(proba >= 0.5)

        st.markdown("---")
        st.subheader("Prediction Result")
        if pred == 1:
            st.success(
                f"✅ This customer is **LIKELY** to purchase the Wellness Tourism Package.\n\n"
                f"Estimated probability: **{proba:.2%}**"
            )
        else:
            st.info(
                f"ℹ️ This customer is **LESS LIKELY** to purchase the Wellness Tourism Package.\n\n"
                f"Estimated probability: **{proba:.2%}**"
            )


if __name__ == "__main__":
    main()
