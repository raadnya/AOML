import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

file_path = "star_classification.csv"  
df = pd.read_csv(file_path)

columns_to_drop = ["obj_ID", "alpha", "delta", "run_ID", "rerun_ID", "cam_col", 
                   "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID"]
df_cleaned = df.drop(columns=columns_to_drop)

categorical_cols = []  
numerical_cols = ["u", "g", "r", "i", "z", "redshift"]

label_encoder = LabelEncoder()
df_cleaned["class"] = label_encoder.fit_transform(df_cleaned["class"])

if categorical_cols:
    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoded_cats = one_hot_encoder.fit_transform(df_cleaned[categorical_cols])
    df_encoded = pd.DataFrame(encoded_cats, columns=one_hot_encoder.get_feature_names_out(categorical_cols))
    df_cleaned = df_cleaned.drop(columns=categorical_cols).reset_index(drop=True)
    df_cleaned = pd.concat([df_cleaned, df_encoded], axis=1)
else:
    one_hot_encoder = None

scaler = StandardScaler()
df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

X = df_cleaned.drop(columns=["class"])
y = df_cleaned["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)

if one_hot_encoder:
    with open("one_hot_encoder.pkl", "wb") as ohe_file:
        pickle.dump(one_hot_encoder, ohe_file)

with open("model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

print("Model, scaler, and encoders saved successfully!")
