import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logs
import tensorflow as tf
from tqdm import tqdm

DATASET_DIR = r"c:\Users\TANMAY\OneDrive\Desktop\PROJECT\MULTIDISPLINARY PROJECT\Review_4\FoodSight-Ai-App\dataset"

def verify_images():
    print(f"Scanning for corrupt images in: {DATASET_DIR}")
    
    # Get all potential image files
    image_files = []
    # Use glob to find files recursively. Note: glob with recursive=True and ** requires python 3.5+
    # Adjust pattern to match what is actually in the dataset
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
        # Case insensitive search would be better but glob doesn't support it directly in older versions easily
        # For now, assume extensions are lowercase or handled
        found = glob.glob(os.path.join(DATASET_DIR, '**', ext), recursive=True)
        image_files.extend(found)
        # Also try uppercase extensions just in case
        found_upper = glob.glob(os.path.join(DATASET_DIR, '**', ext.upper()), recursive=True)
        image_files.extend(found_upper)
        
    print(f"Found {len(image_files)} files to check.")
    
    corrupt_count = 0
    deleted_count = 0
    
    for file_path in tqdm(image_files):
        try:
            # 1. Check file size
            if os.path.getsize(file_path) == 0:
                print(f"[ZERO-BYTE] Deleting: {file_path}")
                os.remove(file_path)
                deleted_count += 1
                continue

            # 2. Check TensorFlow decoding (The ultimate test for training)
            try:
                img_bytes = tf.io.read_file(file_path)
                # Try to decode as image. This handles jpg, png, bmp, gif etc.
                # expand_animations=False is for GIFs to return 3D tensor, consistent with others
                img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
                # Force execution to ensure it's valid
                _ = img.numpy() 
            except Exception as e:
                print(f"[TF-CORRUPT] Deleting: {file_path} - Error: {e}")
                os.remove(file_path)
                deleted_count += 1
                continue

        except Exception as e:
            print(f"[ERROR] Could not process {file_path}: {e}")
    
    print("-" * 50)
    print(f"Scan complete. Found and removed {deleted_count} corrupt files.")

if __name__ == "__main__":
    verify_images()
