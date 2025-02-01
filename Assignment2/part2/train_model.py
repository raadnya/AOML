import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

file_path = "housing.csv"  
df = pd.read_csv(file_path)

df = df.dropna()

X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

n_bins = int(1 + np.log2(len(y)))  
y_binned = pd.cut(y, bins=n_bins, labels=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_binned, random_state=42
)

scaler = StandardScaler()
ohe = OneHotEncoder(handle_unknown="ignore")

X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_train_cat = ohe.fit_transform(X_train[categorical_cols]).toarray()

X_train_transformed = np.hstack([X_train_num, X_train_cat])
X_test_num = scaler.transform(X_test[numerical_cols])
X_test_cat = ohe.transform(X_test[categorical_cols]).toarray()
X_test_transformed = np.hstack([X_test_num, X_test_cat])

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

ridge.fit(X_train_transformed, y_train)
lasso.fit(X_train_transformed, y_train)

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = math.sqrt(mse)
    return mae, mse, rmse

ridge_preds = ridge.predict(X_test_transformed)
lasso_preds = lasso.predict(X_test_transformed)

ridge_metrics = evaluate_model(y_test, ridge_preds)
lasso_metrics = evaluate_model(y_test, lasso_preds)

print("Ridge Regression:")
print(f"MAE: {ridge_metrics[0]:.2f}, MSE: {ridge_metrics[1]:.2f}, RMSE: {ridge_metrics[2]:.2f}")

print("\nLasso Regression:")
print(f"MAE: {lasso_metrics[0]:.2f}, MSE: {lasso_metrics[1]:.2f}, RMSE: {lasso_metrics[2]:.2f}")

with open("ridge_model.pkl", "wb") as file:
    pickle.dump(ridge, file)

with open("lasso_model.pkl", "wb") as file:
    pickle.dump(lasso, file)

with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

with open("one_hot_encoder.pkl", "wb") as file:
    pickle.dump(ohe, file)

print("Models, scaler, and encoder saved successfully!")
