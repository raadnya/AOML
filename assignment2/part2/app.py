import streamlit as st
import pickle
import pandas as pd
import numpy as np

with open("ridge_model.pkl", "rb") as file:
    ridge_model = pickle.load(file)

with open("lasso_model.pkl", "rb") as file:
    lasso_model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("one_hot_encoder.pkl", "rb") as file:
    ohe = pickle.load(file)

numerical_cols = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income"
]
categorical_cols = ["ocean_proximity"]

st.title("California Housing Price Predictor")

st.sidebar.header("Input Features")
longitude = st.sidebar.number_input("Longitude", value=-122.23)
latitude = st.sidebar.number_input("Latitude", value=37.88)
housing_median_age = st.sidebar.number_input("Housing Median Age", value=41)
total_rooms = st.sidebar.number_input("Total Rooms", value=880)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", value=129)
population = st.sidebar.number_input("Population", value=322)
households = st.sidebar.number_input("Households", value=126)
median_income = st.sidebar.number_input("Median Income", value=8.3252)
ocean_proximity = st.sidebar.selectbox("Ocean Proximity", options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

if st.sidebar.button("Predict"):
    features = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }
    
    input_df = pd.DataFrame([features])
    input_num = scaler.transform(input_df[numerical_cols])
    input_cat = ohe.transform(input_df[categorical_cols]).toarray()
    input_transformed = np.hstack([input_num, input_cat])

    ridge_prediction = ridge_model.predict(input_transformed)[0]
    lasso_prediction = lasso_model.predict(input_transformed)[0]

    st.write(f"**Predicted House Value (Ridge):** ${ridge_prediction:.2f}")
    st.write(f"**Predicted House Value (Lasso):** ${lasso_prediction:.2f}")
