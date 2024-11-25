import os
import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tensorflow.keras.applications import ResNet50
from scripts.parse_csv_to_dict import parse_csv_to_dict
from scripts.link_images_to_scores import link_images_to_scores
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import joblib
import json

def extract_image_features(X, batch_size=32):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(512, 512, 3))
    X = X.astype('float32') / 255.0  # Normalize pixel values to [0, 1]

    # Use predict with batch_size to avoid memory issues
    features = base_model.predict(X, batch_size=batch_size)
    return features

def load_data(image_dir_lst, csv_lst, batch_size=32):
    X, y, extracted_feature_lst = [], [], []

    for image_dir, csv_file in zip(image_dir_lst, csv_lst):
        csv_dict = parse_csv_to_dict(csv_file)

        # Process and load the images in batches
        for X_batch, y_batch, features_batch in link_images_to_scores(image_dir, csv_dict, batch_size):
            X.extend(X_batch)
            y.extend(y_batch)
            extracted_feature_lst.extend(features_batch)

            # Free memory after processing each batch
            del X_batch, y_batch, features_batch
            gc.collect()

    # Convert the accumulated data into numpy arrays
    X = np.array(X).astype('float32')
    y = np.array(y)
    extracted_feature_lst = np.array(extracted_feature_lst)

    return X, y, extracted_feature_lst



def ensure_rgb(X):
    """
    Ensures the input images have 3 channels (RGB).
    If the image is grayscale (single-channel), it duplicates the channel to create an RGB image.
    """
    if len(X.shape) == 3:  # If the image is grayscale with shape (num_images, height, width)
        X = np.expand_dims(X, axis=-1)  # Add a channel dimension (num_images, height, width, 1)
    
    if X.shape[-1] == 1:  # If the image has 1 channel, duplicate it to create 3 channels (RGB)
        X = np.repeat(X, 3, axis=-1)
    
    return X



# Hyperparameter search space for XGBClassifier
param_dist = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_lambda': [0.5, 1, 1.5],
}

def cross_validate_and_tune_model(X, y, extracted_features, use_label_features, n_splits=5):
    if use_label_features:
        print("Extracting image features using ResNet...")
        X_features = extract_image_features(X)  # Extract features from images using ResNet
        combined_features = np.concatenate([X_features, extracted_features], axis=1)  # Combine image features and manual features
    else:
        combined_features = extracted_features

    y -= 1  # Adjust target labels if necessary 

    # Standardize the feature set
    best_scaler = None

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store best models and accuracies for each random state
    best_models = {
        'peeling': None,
        'contamination': None,
        'density': None
    }
    best_accuracies = {
        'peeling': 0,
        'contamination': 0,
        'density': 0
    }

    # random_states = [123, 456, 789, 66]
    random_states = [76]
    
    for state in random_states:
        print(f"Tuning hyperparameters with random_state={state}...")

        # Prepare individual models for peeling, contamination, and density
        peeling_model = XGBClassifier(objective='multi:softmax', num_class=3)
        contamination_model = XGBClassifier(objective='multi:softmax', num_class=3)
        density_model = XGBClassifier(objective='multi:softmax', num_class=3)
        
        # Randomized search for hyperparameter tuning
        peeling_search = RandomizedSearchCV(peeling_model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, random_state=state)
        contamination_search = RandomizedSearchCV(contamination_model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, random_state=state)
        density_search = RandomizedSearchCV(density_model, param_distributions=param_dist, n_iter=10, scoring='accuracy', cv=3, random_state=state)

        # Accumulate accuracies for all folds
        peeling_accuracies, contamination_accuracies, density_accuracies = [], [], []

        # Now evaluate each model on stratified KFold test set
        for train_idx, test_idx in skf.split(combined_features, y[:, 0]):
            X_train, X_test = combined_features[train_idx], combined_features[test_idx]
            y_train_peeling, y_test_peeling = y[train_idx, 0], y[test_idx, 0]
            y_train_contamination, y_test_contamination = y[train_idx, 1], y[test_idx, 1]
            y_train_density, y_test_density = y[train_idx, 2], y[test_idx, 2]

            # Fit the scaler only on the training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit the models using the train data in this fold
            peeling_search.fit(X_train_scaled, y_train_peeling)
            contamination_search.fit(X_train_scaled, y_train_contamination)
            density_search.fit(X_train_scaled, y_train_density)

            # Best models after search
            peeling_best_model = peeling_search.best_estimator_
            contamination_best_model = contamination_search.best_estimator_
            density_best_model = density_search.best_estimator_

            # Make predictions on the test data of this fold
            peeling_pred = peeling_best_model.predict(X_test_scaled)
            contamination_pred = contamination_best_model.predict(X_test_scaled)
            density_pred = density_best_model.predict(X_test_scaled)

            # Calculate accuracies for this fold
            peeling_accuracy = accuracy_score(y_test_peeling, peeling_pred)
            contamination_accuracy = accuracy_score(y_test_contamination, contamination_pred)
            density_accuracy = accuracy_score(y_test_density, density_pred)
            print("actual density: ")
            print(str(y_test_density))

            print("predicted density: ")
            print(str(density_pred))

            # Append fold accuracy to respective lists
            peeling_accuracies.append(peeling_accuracy)
            contamination_accuracies.append(contamination_accuracy)
            density_accuracies.append(density_accuracy)

        # Calculate mean accuracies across all folds for the current random state
        mean_peeling_accuracy = np.mean(peeling_accuracies)
        mean_contamination_accuracy = np.mean(contamination_accuracies)
        mean_density_accuracy = np.mean(density_accuracies)

        print(f"Mean Peeling Accuracy with random_state={state}: {mean_peeling_accuracy}")
        print(f"Mean Contamination Accuracy with random_state={state}: {mean_contamination_accuracy}")
        print(f"Mean Density Accuracy with random_state={state}: {mean_density_accuracy}")

        # Track the best models
        if mean_peeling_accuracy > best_accuracies['peeling']:
            best_accuracies['peeling'] = mean_peeling_accuracy
            best_models['peeling'] = peeling_best_model
            best_scaler = scaler

        if mean_contamination_accuracy > best_accuracies['contamination']:
            best_accuracies['contamination'] = mean_contamination_accuracy
            best_models['contamination'] = contamination_best_model
            best_scaler = scaler

        if mean_density_accuracy > best_accuracies['density']:
            best_accuracies['density'] = mean_density_accuracy
            best_models['density'] = density_best_model
            best_scaler = scaler

    # After testing with all random states, print the best models
    print(f"Best Peeling Accuracy: {best_accuracies['peeling']}")
    print(f"Best Contamination Accuracy: {best_accuracies['contamination']}")
    print(f"Best Density Accuracy: {best_accuracies['density']}")

    return best_models['peeling'], best_models['contamination'], best_models['density'], best_scaler

# Ensure the 'models' directory exists
def ensure_models_folder_exists(directory="models"):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Save model parameters to JSON
def save_model_params_to_json(model, filename, directory="models"):
    ensure_models_folder_exists(directory)
    model_params = model.get_params()
    filepath = os.path.join(directory, filename)
    with open(filepath, 'w') as json_file:
        json.dump(model_params, json_file)

# Save the full XGBClassifier model to a JSON-compatible format
def save_full_model_to_json(model, filename, directory="models"):
    ensure_models_folder_exists(directory)
    filepath = os.path.join(directory, filename)
    model.save_model(filepath)

def save_scaler(scaler, filename, directory="models"):
    ensure_models_folder_exists(directory)
    filepath = os.path.join(directory, filename)
    joblib.dump(scaler, filepath)

if __name__ == "__main__":
    # Load the dataset
    rounds = ['round09']  

    image_dir_lst = [f'train/{round}_images' for round in rounds]
    csv_lst = [f'train/scoring_{round}.csv' for round in rounds]

    X_path = f'train/X_{"_".join(rounds)}.npy'
    y_path = f'train/y_{"_".join(rounds)}.npy'
    features_path = f'train/features_{"_".join(rounds)}.npy'
    # Load data
    if os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(features_path):
        print("Loading labeled images and scores from disk...")
        X = np.load(X_path)

        X_rgb = ensure_rgb(X)
        y = np.load(y_path)
        extracted_feature_lst = np.load(features_path)
        
        # Filter out instances where y[i][3] == 1
        filtered_indices = [i for i in range(len(y)) if y[i][3] != 1]

        # Use the filtered indices to filter X_rgb, y, and extracted_feature_lst
        X_rgb_filtered = X_rgb[filtered_indices]
        y_filtered = y[filtered_indices]
        extracted_feature_lst_filtered = extracted_feature_lst[filtered_indices]

        #peeling_model, contamination_model, density_model, scaler = train_xgboost(X_rgb, y, extracted_feature_lst, True)
        peeling_model, contamination_model, density_model, scaler = cross_validate_and_tune_model(X_rgb_filtered, y_filtered, extracted_feature_lst_filtered, True)

        # save_model_params_to_json(peeling_model, "peeling_model_params.json")
        # save_model_params_to_json(contamination_model, "contamination_model_params.json")
        # save_model_params_to_json(density_model, "density_model_params.json")

        save_full_model_to_json(peeling_model, "peeling_model3.json")
        save_full_model_to_json(contamination_model, "contamination_model3.json")
        save_full_model_to_json(density_model, "density_model3.json")
        save_scaler(scaler, "scaler3.pkl")
    else:
        X, y, extracted_feature_lst = load_data(image_dir_lst, csv_lst)
        np.save(X_path, X)
        np.save(y_path, y)
        np.save(features_path, extracted_feature_lst)
        #peeling_model, contamination_model, density_model, scaler = train_xgboost(ensure_rgb(X), y, extracted_feature_lst, True)
        X_rgb = ensure_rgb(X)
        
        # Filter out instances where y[i][3] == 1
        filtered_indices = [i for i in range(len(y)) if y[i][3] != 1]

        # Use the filtered indices to filter X_rgb, y, and extracted_feature_lst
        X_rgb_filtered = X_rgb[filtered_indices]
        y_filtered = y[filtered_indices]
        extracted_feature_lst_filtered = extracted_feature_lst[filtered_indices]



        #peeling_model, contamination_model, density_model, scaler = train_xgboost(X_rgb, y, extracted_feature_lst, True)
        peeling_model, contamination_model, density_model, scaler = cross_validate_and_tune_model(X_rgb_filtered, y_filtered, extracted_feature_lst_filtered, True)

        # save_model_params_to_json(peeling_model, "peeling_model_params.json")
        # save_model_params_to_json(contamination_model, "contamination_model_params.json")
        # save_model_params_to_json(density_model, "density_model_params.json")

        save_full_model_to_json(peeling_model, "peeling_model3.json")
        save_full_model_to_json(contamination_model, "contamination_model3.json")
        save_full_model_to_json(density_model, "density_model3.json")
        save_scaler(scaler, "scaler3.pkl")

        
