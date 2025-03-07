import pickle
import numpy as np
import pandas as pd

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

def predict_house_value(features, model="ridge"):
    """
    Predict house value using Ridge or Lasso model.
    :param features: Dictionary of input features
    :param model: "ridge" or "lasso"
    :return: Predicted house value
    """
    input_df = pd.DataFrame([features])

    input_num = scaler.transform(input_df[numerical_cols])
    input_cat = ohe.transform(input_df[categorical_cols]).toarray()
    input_transformed = np.hstack([input_num, input_cat])

    if model == "ridge":
        prediction = ridge_model.predict(input_transformed)
    elif model == "lasso":
        prediction = lasso_model.predict(input_transformed)
    else:
        raise ValueError("Model must be 'ridge' or 'lasso'")
    
    return prediction[0]

if __name__ == "__main__":
    test_features = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41,
        "total_rooms": 880,
        "total_bedrooms": 129,
        "population": 322,
        "households": 126,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"
    }
    print("Predicted House Value (Ridge):", predict_house_value(test_features, model="ridge"))
    print("Predicted House Value (Lasso):", predict_house_value(test_features, model="lasso"))
