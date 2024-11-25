import os
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tensorflow.keras.applications import ResNet50
from scripts.link_images_to_scores import link_images_to_scores_test
import tensorflow as tf
import joblib

def extract_image_features(X, batch_size=32):
    """
    Extracts deep features from the images using ResNet50 in batches.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(512, 512, 3))
    X = X.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

    # Use predict with batch_size to avoid memory issues
    features = base_model.predict(X, batch_size=batch_size)
    return features

def load_data(image_dir_lst, batch_size=32):
    """
    Loads images and the corresponding manual features in batches.
    """
    X, extracted_feature_lst, image_names = [], [], []

    for image_dir in image_dir_lst:
        # Get batches of data from link_images_to_scores_test
        for X_batch, extracted_feature_lst_batch, image_names_batch in link_images_to_scores_test(image_dir, batch_size):
            X.extend(X_batch)
            extracted_feature_lst.extend(extracted_feature_lst_batch)
            image_names.extend(image_names_batch)  # Collect image names

            # Free memory after processing each batch
            del X_batch, extracted_feature_lst_batch, image_names_batch
            gc.collect()

    X = np.array(X).astype('float32')
    extracted_feature_lst = np.array(extracted_feature_lst)

    return X, extracted_feature_lst, image_names

def ensure_rgb(X):
    """
    Ensures the input images have 3 channels (RGB).
    """
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=-1)  # Add a channel dimension
    
    if X.shape[-1] == 1:
        X = np.repeat(X, 3, axis=-1)
    
    return X

def test_xgboost(image_dir_lst, peeling_model_path, contamination_model_path, density_model_path, scaler_path, output_csv, batch_size=32):
    """
    Loads pre-trained models, extracts features from the images in batches, makes predictions, and saves them to a CSV file.
    """
    # Load pre-trained models
    peeling_model = XGBClassifier()
    peeling_model.load_model(peeling_model_path)

    contamination_model = XGBClassifier()
    contamination_model.load_model(contamination_model_path)

    density_model = XGBClassifier()
    density_model.load_model(density_model_path)

    # Load the saved scaler
    scaler = joblib.load(scaler_path)

    # Load the dataset in batches
    X, extracted_feature_lst, image_names = load_data(image_dir_lst, batch_size)

    X_rgb = ensure_rgb(X)
    
    print("Extracting image features using ResNet...")
    X_features = extract_image_features(X_rgb, batch_size=batch_size)  # Extract features from images using ResNet

    # Combine image features and manually extracted features
    combined_features = np.concatenate([X_features, extracted_feature_lst], axis=1)

    # Scale the combined features
    combined_features_scaled = scaler.transform(combined_features)

    # Prepare to store results
    results = []

    # Make predictions using the pre-trained models in batches
    for i in range(len(combined_features_scaled)):
        peeling_pred_class = peeling_model.predict([combined_features_scaled[i]])[0] + 1
        contamination_pred_class = contamination_model.predict([combined_features_scaled[i]])[0] + 1
        density_pred_class = density_model.predict([combined_features_scaled[i]])[0] + 1

        # Determine if the sample is Dead/Empty based on extracted features
        isDead = bool(extracted_feature_lst[i][0])  # extracted_feature_lst[i][0] corresponds to Dead/Empty classification

        # Append the result, using the image name
        results.append((image_names[i], peeling_pred_class, contamination_pred_class, density_pred_class, isDead))

    # Save the predictions to a CSV file
    if results:
        df = pd.DataFrame(results, columns=['Image', 'Peeling', 'Contamination', 'Cell Density', 'Dead/Empty'])
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    # Define the image directories and CSV files to load

    rounds = ['round10']  
    image_dir_lst = [f'test/{round}_images' for round in rounds] 

    
    # Define paths to the pre-trained models and scaler
    peeling_model_path = 'models/peeling_model3.json'
    contamination_model_path = 'models/contamination_model3.json'
    density_model_path = 'models/density_model3.json'
    scaler_path = 'models/scaler3.pkl'

    output_csv = f'xgb_predictions_{rounds[0]}.csv'



    # Run the test function and save predictions to CSV
    test_xgboost(image_dir_lst, peeling_model_path, contamination_model_path, density_model_path, scaler_path, output_csv)

