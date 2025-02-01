import pickle
import numpy as np
import pandas as pd

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

try:
    with open("one_hot_encoder.pkl", "rb") as ohe_file:
        one_hot_encoder = pickle.load(ohe_file)
except FileNotFoundError:
    one_hot_encoder = None

with open("model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

feature_names = ["u", "g", "r", "i", "z", "redshift"]

def predict_stellar_class(features):
    """
    Predict the stellar class given input features.
    :param features: List of input features [u, g, r, i, z, redshift]
    :return: Predicted class name (Galaxy, Star, QSO)
    """
    features_df = pd.DataFrame([features], columns=feature_names)

    features_scaled = scaler.transform(features_df)

    if one_hot_encoder:
        encoded_cats = one_hot_encoder.transform([[]])  
        features_scaled = np.concatenate([features_scaled, encoded_cats], axis=1)

    prediction = clf.predict(features_scaled)
    class_name = label_encoder.inverse_transform(prediction)[0]
    return class_name

if __name__ == "__main__":
    test_input = [23.87882, 22.27530, 20.39501, 19.16573, 18.79371, 0.634794]  
    print("Predicted Class:", predict_stellar_class(test_input))
