import pandas as pd
import numpy as np
import pickle
import os

# Load saved artifacts
def load_artifacts():
    print("Loading saved artifacts...")
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded.")

    # Check if encoder.pkl exists
    encoder_path = "models/encoder.pkl"
    if os.path.exists(encoder_path):
        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)
        print("Encoder loaded.")
    else:
        encoder = None  # No encoder needed for Decision Tree
        print("Encoder not found. Skipping encoder loading.")

    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Best model loaded.")
    return scaler, encoder, model

# Preprocess a single sample
def preprocess_sample(sample, numeric_features, categorical_features, scaler, encoder, model_type):
    # Scale numeric features
    sample_numeric = scaler.transform(sample[numeric_features])

    if model_type == "KNN" and encoder is not None:
        # One-hot encode categorical features for KNN
        sample_cat = encoder.transform(sample[categorical_features]).toarray()
        sample_preprocessed = np.hstack((sample_numeric, sample_cat))
    else:  # Decision Tree
        # Label encode categorical features for Decision Tree
        label_encoded_cat = sample[categorical_features].replace(
            {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
        ).astype(int).values
        sample_preprocessed = np.hstack((sample_numeric, label_encoded_cat.reshape(-1, 1)))

    return sample_preprocessed

# Predict the class for a single sample
def predict(sample, numeric_features, categorical_features, scaler, encoder, model):
    # Determine the type of model (KNN or Decision Tree)
    model_type = "KNN" if encoder is not None else "Decision Tree"
    print(f"Using {model_type} for inference.")

    # Preprocess the sample
    sample_preprocessed = preprocess_sample(sample, numeric_features, categorical_features, scaler, encoder, model_type)

    # Predict the class
    prediction = model.predict(sample_preprocessed)
    return prediction[0]  # Return the predicted class

# Main function to demonstrate inference
if __name__ == "__main__":
    # Load dataset
    filepath = "data/diabetes.csv"
    df = pd.read_csv(filepath)

    # Add BMI categories
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

    # Define numeric and categorical features
    numeric_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                        'BMI', 'DiabetesPedigreeFunction', 'Age']
    categorical_features = ['BMI_category']

    # Load saved artifacts
    scaler, encoder, model = load_artifacts()

    # Select 5 samples from the dataset for testing
    sample_data = df.sample(5, random_state=42).reset_index(drop=True)
    print("\nSelected samples for inference:")
    print(sample_data)

    # Predict for each sample and store predictions
    predictions = []
    for i in range(len(sample_data)):
        sample = sample_data.iloc[i:i+1]  # Select one sample
        prediction = predict(sample, numeric_features, categorical_features, scaler, encoder, model)
        predictions.append(prediction)

    # Add predictions to the DataFrame
    sample_data['Predicted Outcome'] = predictions

    # Print the updated DataFrame
    print("\nInference Results:")
    print(sample_data)
