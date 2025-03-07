import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import pickle


filepath = "data/diabetes.csv"
df = pd.read_csv(filepath)
def add_bmi_category(df):
    def get_bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 24.9:
            return 'Normal'
        elif 25 <= bmi < 29.9:
            return 'Overweight'
        else:
            return 'Obese'
    
    
    df['BMI_category'] = df['BMI'].apply(get_bmi_category)
    return df


df = add_bmi_category(df)




def preprocess_data(df):
    print("Starting preprocessing...")

    
    numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                        'BMI', 'DiabetesPedigreeFunction', 'Age']
    categorical_features = ['BMI_category']

    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']

    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")

    
    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(X_train[numeric_features])
    X_val_numeric = scaler.transform(X_val[numeric_features])
    print("Numeric features scaled.")

    
    encoder = OneHotEncoder()
    X_train_cat_knn = encoder.fit_transform(X_train[categorical_features]).toarray()
    X_val_cat_knn = encoder.transform(X_val[categorical_features]).toarray()
    print("Categorical features encoded for KNN.")

    
    label_encoder = LabelEncoder()
    X_train_cat_tree = label_encoder.fit_transform(X_train[categorical_features].values.ravel())
    X_val_cat_tree = label_encoder.transform(X_val[categorical_features].values.ravel())
    print("Categorical features encoded for Decision Tree.")

    
    X_train_knn = np.hstack((X_train_numeric, X_train_cat_knn))
    X_val_knn = np.hstack((X_val_numeric, X_val_cat_knn))

    X_train_tree = pd.concat([pd.DataFrame(X_train_numeric, columns=numeric_features), 
                              pd.DataFrame(X_train_cat_tree, columns=['BMI_category'])], axis=1)
    X_val_tree = pd.concat([pd.DataFrame(X_val_numeric, columns=numeric_features), 
                            pd.DataFrame(X_val_cat_tree, columns=['BMI_category'])], axis=1)

    print("Data ready for both KNN and Decision Tree.")
    return X_train_knn, X_val_knn, X_train_tree, X_val_tree, y_train, y_val, scaler, encoder


X_train_knn, X_val_knn, X_train_tree, X_val_tree, y_train, y_val, scaler, encoder=preprocess_data(df)

def train_models(X_train_knn, X_val_knn, X_train_tree, X_val_tree, y_train, y_val):
    print("Training models...")


    best_knn_f1 = 0
    best_k = None
    best_knn_model = None
    for k in [3, 5, 7]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_knn, y_train)
        y_pred = knn.predict(X_val_knn)
        f1 = f1_score(y_val, y_pred)
        print(f"KNN (k={k}): F1 Score = {f1}")
        if f1 > best_knn_f1:
            best_knn_f1 = f1
            best_k = k
            best_knn_model = knn

    
    best_tree_f1 = 0
    best_depth = None
    best_tree_model = None
    for depth in [3, 5, 7]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train_tree, y_train)
        y_pred = tree.predict(X_val_tree)
        f1 = f1_score(y_val, y_pred)
        print(f"Decision Tree (max_depth={depth}): F1 Score = {f1}")
        if f1 > best_tree_f1:
            best_tree_f1 = f1
            best_depth = depth
            best_tree_model = tree

    
    if best_knn_f1 > best_tree_f1:
        print(f"Best Model: KNN with k={best_k}, F1 Score = {best_knn_f1}")
        return best_knn_model, 'KNN', best_knn_f1
    else:
        print(f"Best Model: Decision Tree with max_depth={best_depth}, F1 Score = {best_tree_f1}")
        return best_tree_model, 'Decision Tree', best_tree_f1

X_train_knn, X_val_knn, X_train_tree, X_val_tree, y_train, y_val, scaler, encoder = preprocess_data(df)
best_model, model_name, best_f1 = train_models(X_train_knn, X_val_knn, X_train_tree, X_val_tree, y_train, y_val)


print(f"Saving the best model ({model_name}) and preprocessing tools...")


import os
os.makedirs("models", exist_ok=True)


with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved.")

if model_name == "KNN":
    with open("models/encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    print("Encoder saved.")


with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print(f"Best model ({model_name}) saved successfully.")