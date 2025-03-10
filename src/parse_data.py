import os
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from tqdm import tqdm  # Use tqdm.notebook for Jupyter/Kaggle environments


# Image processing functions
def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles
    """
    # Calculate percentiles
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)

    # Clip the data to the percentile range
    clipped_data = np.clip(slice_data, p2, p98)

    # Normalize to [0, 255] range
    normalized = 255 * (clipped_data - p2) / (p98 - p2)

    return np.uint8(normalized)


def prepare_yolo_dataset(trust=4, train_split=0.8):
    """
    Extract slices containing motors from tomograms and save to YOLO structure with annotations
    """
    # Load the labels CSV
    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))

    # Count total number of motors
    total_motors = labels_df['Number of motors'].sum()
    print(f"Total number of motors in the dataset: {total_motors}")

    # Get unique tomograms that have motors
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].unique()

    print(f"Found {len(unique_tomos)} unique tomograms with motors")

    # Perform the train-val split at the tomogram level (not motor level)
    # This ensures all slices from a single tomogram go to either train or val
    np.random.shuffle(unique_tomos)  # Shuffle the tomograms
    split_idx = int(len(unique_tomos) * train_split)
    train_tomos = unique_tomos[:split_idx]
    val_tomos = unique_tomos[split_idx:]

    print(
        f"Split: {len(train_tomos)} tomograms for training, {len(val_tomos)} tomograms for validation")

    # Function to process a set of tomograms
    def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            # Get all motors for this tomogram
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id,
                     int(motor['Motor axis 0']),
                     int(motor['Motor axis 1']),
                     int(motor['Motor axis 2']),
                     int(motor['Array shape (axis 0)']))
                )

        print(
            f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")

        # Process each motor
        processed_slices = 0

        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
            # Calculate range of slices to include
            z_min = max(0, z_center - trust)
            z_max = min(z_max - 1, z_center + trust)

            # Process each slice in the range
            for z in range(z_min, z_max + 1):
                # Create slice filename
                slice_filename = f"slice_{z:04d}.jpg"

                # Source path for the slice
                src_path = os.path.join(train_dir, tomo_id, slice_filename)

                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} does not exist, skipping.")
                    continue

                # Load and normalize the slice
                img = Image.open(src_path)
                img_array = np.array(img)

                # Normalize the image
                normalized_img = normalize_slice(img_array)

                # Create destination filename (with unique identifier)
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)

                # Save the normalized image
                Image.fromarray(normalized_img).save(dest_path)

                # Get image dimensions
                img_width, img_height = img.size

                # Create YOLO format label
                # YOLO format: <class> <x_center> <y_center> <width> <height>
                # Values are normalized to [0, 1]
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                box_width_norm = BOX_SIZE / img_width
                box_height_norm = BOX_SIZE / img_height

                # Write label file
                label_path = os.path.join(
                    labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(
                        f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")

                processed_slices += 1

        return processed_slices, len(motor_counts)

    # Process training tomograms
    train_slices, train_motors = process_tomogram_set(
        train_tomos, yolo_images_train, yolo_labels_train, "training")

    # Process validation tomograms
    val_slices, val_motors = process_tomogram_set(
        val_tomos, yolo_images_val, yolo_labels_val, "validation")

    # Create YAML configuration file for YOLO
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'}
    }

    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print("Processing Summary:")
    print(
        f"- Train set: {len(train_tomos)} tomograms, {train_motors} motors, {train_slices} slices")
    print(
        f"- Validation set: {len(val_tomos)} tomograms, {val_motors} motors, {val_slices} slices")
    print(f"- Total: {len(train_tomos) + len(val_tomos)} tomograms, {train_motors + val_motors} motors, {train_slices + val_slices} slices")

    # Return summary info
    return {
        "dataset_dir": yolo_dataset_dir,
        "yaml_path": os.path.join(yolo_dataset_dir, 'dataset.yaml'),
        "train_tomograms": len(train_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "train_slices": train_slices,
        "val_slices": val_slices
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Define Kaggle paths
    data_path = "data/"
    train_dir = os.path.join(data_path, "train")

    # Define YOLO dataset structure
    yolo_dataset_dir = "data/yolo_dataset"
    yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
    yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
    yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
    yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

    # Create directories
    for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
        os.makedirs(dir_path, exist_ok=True)

    # Define constants
    # Number of slices above and below center slice (total 2*TRUST + 1 slices)
    TRUST = 4
    BOX_SIZE = 24  # Bounding box size for annotations (in pixels)
    TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation

    # Run the preprocessing
    summary = prepare_yolo_dataset(TRUST)
    print("Preprocessing Complete:")
    print(
        f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
    print(
        f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
    print(f"- Dataset directory: {summary['dataset_dir']}")
    print(f"- YAML configuration: {summary['yaml_path']}")
    print(f"Ready for YOLO training!")
