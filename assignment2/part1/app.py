import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

with open("model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("label_encoder.pkl", "rb") as encoder_file:
    label_encoder = pickle.load(encoder_file)

try:
    with open("one_hot_encoder.pkl", "rb") as ohe_file:
        one_hot_encoder = pickle.load(ohe_file)
except FileNotFoundError:
    one_hot_encoder = None

st.title("Stellar Classification App")

st.sidebar.header("Enter Stellar Features:")
u = st.sidebar.number_input("u-band magnitude", value=23.0)
g = st.sidebar.number_input("g-band magnitude", value=22.0)
r = st.sidebar.number_input("r-band magnitude", value=20.0)
i = st.sidebar.number_input("i-band magnitude", value=19.0)
z = st.sidebar.number_input("z-band magnitude", value=18.0)
redshift = st.sidebar.number_input("Redshift", value=0.5)

if st.sidebar.button("Predict"):
    features = np.array([u, g, r, i, z, redshift]).reshape(1, -1)
    features = scaler.transform(features)

    if one_hot_encoder:
        encoded_cats = one_hot_encoder.transform([[]])  
        features = np.concatenate([features, encoded_cats], axis=1)

    prediction = clf.predict(features)
    class_name = label_encoder.inverse_transform(prediction)[0]
    st.sidebar.success(f"Predicted Class: **{class_name}**")

df = pd.read_csv("star_classification.csv")

df_cleaned = df.drop(columns=["obj_ID", "alpha", "delta", "run_ID", "rerun_ID", "cam_col", 
                              "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID"])
df_cleaned["class"] = label_encoder.transform(df_cleaned["class"])
X_test = df_cleaned.drop(columns=["class"])
y_test = df_cleaned["class"]
y_pred = clf.predict(X_test)

st.subheader("Classification Report")
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
st.pyplot(fig)
