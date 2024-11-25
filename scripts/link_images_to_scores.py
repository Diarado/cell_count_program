import os
from scripts.preprocess import preprocess_image
import gc

def link_images_to_scores(image_dir, csv_dict, batch_size=32):
    X_batch, y_batch, extracted_feature_batch = [], [], []  # To hold batch data

    # Iterate through the files in the image directory
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.tif'):
                image_path = os.path.join(root, file)

                # Extract useful information from the file name (similar to your original code)
                info = file.split('_')[-1]  # Extract useful info (e.g., H07fld04)
                plate_num = info.split('fld')[0]  # Extract plate number (e.g., H07)
                fld_num_str = info.split('fld')[-1]  # Extract field number with extension (e.g., 04.tif)
                fld_num = int(fld_num_str.split('.')[0]) - 1  # Remove extension, convert to zero-indexed integer

                rep_folder = os.path.basename(root)  # Extract replicate number (e.g., rep1)
                plate_info = f"{rep_folder}_{plate_num}"  # Combine to form plate_info (e.g., rep1_H07)

                if plate_info in csv_dict:
                    scores = csv_dict[plate_info][fld_num]

                    # Preprocess the image
                    processed_image, extracted_features = preprocess_image(image_path)

                    # Append to batch
                    X_batch.append(processed_image)
                    y_batch.append(scores)
                    extracted_feature_batch.append(extracted_features)

                    # Check if the batch size is reached
                    if len(X_batch) == batch_size:
                        # Yield the batch
                        yield X_batch, y_batch, extracted_feature_batch

                        # Clear the batch lists to prepare for the next batch
                        X_batch, y_batch, extracted_feature_batch = [], [], []
                        gc.collect()  # Collect garbage to free memory

                else:
                    print(f"Plate info {plate_info} not found in CSV dictionary.")

    # If there are leftover images that didn't fill a full batch, yield them
    if len(X_batch) > 0:
        yield X_batch, y_batch, extracted_feature_batch
        gc.collect()

    
def link_images_to_scores_test(image_dir, batch_size=32):
    """
    Processes the images in batches and yields batches of processed images,
    manually extracted features, and image names.
    """
    X_batch, extracted_feature_batch, image_names_batch = [], [], []

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.tif'):
                image_path = os.path.join(root, file)

                # Extract useful information from the file name
                info = file.split('_')[-1]  # Extract useful info (e.g., H07fld04)
                plate_num = info.split('fld')[0]  # Extract plate number (e.g., H07)
                fld_num_str = info.split('fld')[-1]  # Extract field number with extension (e.g., 04.tif)
                fld_num = int(fld_num_str.split('.')[0])  # Remove extension, convert to zero-indexed integer

                rep_folder = os.path.basename(root)  # Extract replicate number (e.g., rep1)
                plate_info = f"{rep_folder}_{plate_num}"  # Combine to form plate_info (e.g., rep1_H07)
                image_name = plate_info + '_' + str(fld_num)

                # Preprocess the image
                processed_image, extracted_features = preprocess_image(image_path)

                # Append to batch
                X_batch.append(processed_image)
                extracted_feature_batch.append(extracted_features)
                image_names_batch.append(image_name)

                # Check if the batch size is reached
                if len(X_batch) == batch_size:
                    # Yield the batch
                    yield X_batch, extracted_feature_batch, image_names_batch

                    # Clear the batch lists to prepare for the next batch
                    X_batch, extracted_feature_batch, image_names_batch = [], [], []
                    gc.collect()  # Collect garbage to free memory

    # If there are leftover images that didn't fill a full batch, yield them
    if len(X_batch) > 0:
        yield X_batch, extracted_feature_batch, image_names_batch
        gc.collect()
