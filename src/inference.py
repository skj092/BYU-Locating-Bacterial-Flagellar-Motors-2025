import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import cv2
from tqdm.notebook import tqdm
from ultralytics import YOLO
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths
data_path = "data/"
test_dir = os.path.join(data_path, "test")
submission_path = "submission.csv"

# Model path - adjust if your best model is saved in a different location
model_path = "yolov8n.pt"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.45  # Lower threshold to catch more potential motors
MAX_DETECTIONS_PER_TOMO = 3  # Keep track of top N detections per tomogram
NMS_IOU_THRESHOLD = 0.2  # Non-maximum suppression threshold for 3D clustering
CONCENTRATION = 1  # ONLY PROCESS 1/20 slices for fast submission

# GPU profiling context manager


class GPUProfiler:
    def __init__(self, name):
        self.name = name
        self.start_time = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        print(f"[PROFILE] {self.name}: {elapsed:.3f}s")


# Check GPU availability and set up optimizations
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8  # Default batch size, will be adjusted dynamically if GPU available

if device.startswith('cuda'):
    # Set CUDA optimization flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
    torch.backends.cudnn.allow_tf32 = True

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(
        0).total_memory / 1e9  # Convert to GB
    print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")

    # Get available GPU memory and set batch size accordingly
    free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
    # 4 images per GB as rough estimate
    BATCH_SIZE = max(8, min(32, int(free_mem * 4)))
    print(
        f"Dynamic batch size set to {BATCH_SIZE} based on {free_mem:.2f}GB free memory")
else:
    print("GPU not available, using CPU")
    BATCH_SIZE = 4  # Reduce batch size for CPU


def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles for better contrast
    """
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)


def preload_image_batch(file_paths):
    """Preload a batch of images to CPU memory"""
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is None:
            # Try with PIL as fallback
            img = np.array(Image.open(path))
        images.append(img)
    return images


def process_tomogram(tomo_id, model, index=0, total=1):
    """
    Process a single tomogram and return the most confident motor detection
    """
    print(f"Processing tomogram {tomo_id} ({index}/{total})")

    # Get all slice files for this tomogram
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted(
        [f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])

    # Apply CONCENTRATION to reduce the number of slices processed
    # This will process approximately CONCENTRATION fraction of all slices
    selected_indices = np.linspace(
        0, len(slice_files)-1, int(len(slice_files) * CONCENTRATION))
    selected_indices = np.round(selected_indices).astype(int)
    slice_files = [slice_files[i] for i in selected_indices]

    print(
        f"Processing {len(slice_files)} out of {len(os.listdir(tomo_dir))} slices based on CONCENTRATION={CONCENTRATION}")

    # Create a list to store all detections
    all_detections = []

    # Create CUDA streams for parallel processing if using GPU
    if device.startswith('cuda'):
        streams = [torch.cuda.Stream() for _ in range(min(4, BATCH_SIZE))]
    else:
        streams = [None]

    # Variables for preloading
    next_batch_thread = None
    next_batch_images = None

    # Process slices in batches
    for batch_start in range(0, len(slice_files), BATCH_SIZE):
        # Wait for previous preload thread if it exists
        if next_batch_thread is not None:
            next_batch_thread.join()
            next_batch_images = None

        batch_end = min(batch_start + BATCH_SIZE, len(slice_files))
        batch_files = slice_files[batch_start:batch_end]

        # Start preloading next batch
        next_batch_start = batch_end
        next_batch_end = min(next_batch_start + BATCH_SIZE, len(slice_files))
        next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(
            slice_files) else []

        if next_batch_files:
            next_batch_paths = [os.path.join(tomo_dir, f)
                                for f in next_batch_files]
            next_batch_thread = threading.Thread(
                target=preload_image_batch, args=(next_batch_paths,))
            next_batch_thread.start()
        else:
            next_batch_thread = None

        # Split batch across streams for parallel processing
        sub_batches = np.array_split(batch_files, len(streams))
        sub_batch_results = []

        for i, sub_batch in enumerate(sub_batches):
            if len(sub_batch) == 0:
                continue

            stream = streams[i % len(streams)]
            with torch.cuda.stream(stream) if stream and device.startswith('cuda') else nullcontext():
                # Process sub-batch
                sub_batch_paths = [os.path.join(
                    tomo_dir, slice_file) for slice_file in sub_batch]
                sub_batch_slice_nums = [int(slice_file.split('_')[1].split('.')[
                                            0]) for slice_file in sub_batch]

                # Run inference with profiling
                with GPUProfiler(f"Inference batch {i+1}/{len(sub_batches)}"):
                    sub_results = model(sub_batch_paths, verbose=False)

                # Process each result in this sub-batch
                for j, result in enumerate(sub_results):
                    if len(result.boxes) > 0:
                        boxes = result.boxes
                        for box_idx, confidence in enumerate(boxes.conf):
                            if confidence >= CONFIDENCE_THRESHOLD:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = boxes.xyxy[box_idx].cpu(
                                ).numpy()

                                # Calculate center coordinates
                                x_center = (x1 + x2) / 2
                                y_center = (y1 + y2) / 2

                                # Store detection with 3D coordinates
                                all_detections.append({
                                    'z': round(sub_batch_slice_nums[j]),
                                    'y': round(y_center),
                                    'x': round(x_center),
                                    'confidence': float(confidence)
                                })

        # Synchronize streams
        if device.startswith('cuda'):
            torch.cuda.synchronize()

    # Clean up thread if still running
    if next_batch_thread is not None:
        next_batch_thread.join()

    # 3D Non-Maximum Suppression to merge nearby detections across slices
    final_detections = perform_3d_nms(all_detections, NMS_IOU_THRESHOLD)

    # Sort detections by confidence (highest first)
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)

    # If there are no detections, return NA values
    if not final_detections:
        return {
            'tomo_id': tomo_id,
            'Motor axis 0': -1,
            'Motor axis 1': -1,
            'Motor axis 2': -1
        }

    # Take the detection with highest confidence
    best_detection = final_detections[0]

    # Return result with integer coordinates
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': round(best_detection['z']),
        'Motor axis 1': round(best_detection['y']),
        'Motor axis 2': round(best_detection['x'])
    }


def perform_3d_nms(detections, iou_threshold):
    """
    Perform 3D Non-Maximum Suppression on detections to merge nearby motors
    """
    if not detections:
        return []

    # Sort by confidence (highest first)
    detections = sorted(
        detections, key=lambda x: x['confidence'], reverse=True)

    # List to store final detections after NMS
    final_detections = []

    # Define 3D distance function
    def distance_3d(d1, d2):
        return np.sqrt((d1['z'] - d2['z'])**2 +
                       (d1['y'] - d2['y'])**2 +
                       (d1['x'] - d2['x'])**2)

    # Maximum distance threshold (based on box size and slice gap)
    box_size = 24  # Same as annotation box size
    distance_threshold = box_size * iou_threshold

    # Process each detection
    while detections:
        # Take the detection with highest confidence
        best_detection = detections.pop(0)
        final_detections.append(best_detection)

        # Filter out detections that are too close to the best detection
        detections = [d for d in detections if distance_3d(
            d, best_detection) > distance_threshold]

    return final_detections


def debug_image_loading(tomo_id):
    """
    Debug function to check image loading
    """
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted(
        [f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])

    if not slice_files:
        print(f"No image files found in {tomo_dir}")
        return

    print(f"Found {len(slice_files)} image files in {tomo_dir}")
    sample_file = slice_files[len(slice_files)//2]  # Middle slice
    img_path = os.path.join(tomo_dir, sample_file)

    # Try different loading methods
    try:
        # Method 1: PIL
        img_pil = Image.open(img_path)
        img_array_pil = np.array(img_pil)
        print(
            f"PIL Image shape: {img_array_pil.shape}, dtype: {img_array_pil.dtype}")

        # Method 2: OpenCV
        img_cv2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print(f"OpenCV Image shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")

        # Method 3: Convert to RGB
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print(
            f"OpenCV RGB Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")

        print("Image loading successful!")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

    # Also test with YOLO's built-in loader
    try:
        test_model = YOLO(model_path)
        test_results = test_model([img_path], verbose=False)
        print("YOLO model successfully processed the test image")
    except Exception as e:
        print(f"Error with YOLO processing: {e}")


def generate_submission():
    """
    Main function to generate the submission file
    """
    # Get list of test tomograms
    test_tomos = sorted([d for d in os.listdir(test_dir)
                        if os.path.isdir(os.path.join(test_dir, d))])
    total_tomos = len(test_tomos)

    print(f"Found {total_tomos} tomograms in test directory")

    # Debug image loading for the first tomogram
    if test_tomos:
        debug_image_loading(test_tomos[0])

    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize model once outside the processing loop
    print(f"Loading YOLO model from {model_path}")
    model = YOLO(model_path)
    model.to(device)

    # Additional optimizations for inference
    if device.startswith('cuda'):
        # Fuse conv and bn layers for faster inference
        model.fuse()

        # Enable model half precision (FP16) if on compatible GPU
        if torch.cuda.get_device_capability(0)[0] >= 7:  # Volta or newer
            model.model.half()
            print("Using half precision (FP16) for inference")

    # Process tomograms with parallelization
    results = []
    motors_found = 0

    # Using ThreadPoolExecutor with max_workers=1 since each worker uses the GPU already
    # and we're parallelizing within each tomogram processing
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_tomo = {}

        # Submit all tomograms for processing
        for i, tomo_id in enumerate(test_tomos, 1):
            future = executor.submit(
                process_tomogram, tomo_id, model, i, total_tomos)
            future_to_tomo[future] = tomo_id

        # Process completed futures as they complete
        for future in future_to_tomo:
            tomo_id = future_to_tomo[future]
            try:
                # Clear CUDA cache between tomograms
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                result = future.result()
                results.append(result)

                # Update motors found count
                has_motor = not pd.isna(result['Motor axis 0'])
                if has_motor:
                    motors_found += 1
                    print(f"Motor found in {tomo_id} at position: "
                          f"z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
                else:
                    print(f"No motor detected in {tomo_id}")

                print(
                    f"Current detection rate: {motors_found}/{len(results)} ({motors_found/len(results)*100:.1f}%)")

            except Exception as e:
                print(f"Error processing {tomo_id}: {e}")
                # Create a default entry for failed tomograms
                results.append({
                    'tomo_id': tomo_id,
                    'Motor axis 0': -1,
                    'Motor axis 1': -1,
                    'Motor axis 2': -1
                })

    # Create submission dataframe
    submission_df = pd.DataFrame(results)

    # Ensure proper column order
    submission_df = submission_df[[
        'tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]

    # Save the submission file
    submission_df.to_csv(submission_path, index=False)

    print(f"\nSubmission complete!")
    print(
        f"Motors detected: {motors_found}/{total_tomos} ({motors_found/total_tomos*100:.1f}%)")
    print(f"Submission saved to: {submission_path}")

    # Display first few rows of submission
    print("\nSubmission preview:")
    print(submission_df.head())

    return submission_df


# Run the submission pipeline
if __name__ == "__main__":
    # Time entire process
    start_time = time.time()

    # Generate submission
    submission = generate_submission()

    # Print total execution time
    elapsed = time.time() - start_time
    print(
        f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
