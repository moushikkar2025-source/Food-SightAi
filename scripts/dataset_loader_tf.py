import tensorflow as tf
import os
import pathlib

def load_dataset(base_dir, img_height=224, img_width=224, batch_size=32):
    """
    Loads training, validation, and test datasets from distinct directories.
    Handles 100 classes for FoodSight AI.
    """
    base_path = pathlib.Path(base_dir)
    train_dir = base_path / "train"
    val_dir = base_path / "validation"
    test_dir = base_path / "test"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found at {train_dir}")

    print(f"Loading datasets from: {base_path}")
    
    # Common parameters for image_dataset_from_directory
    loader_params = {
        "image_size": (img_height, img_width),
        "batch_size": batch_size,
        "label_mode": 'categorical'
    }

    # Training set
    train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, **loader_params)
    class_names = train_ds.class_names

    # Validation set (Use physical dir if exists, else error)
    if val_dir.exists():
        val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, **loader_params)
    else:
        print("Warning: Validation directory not found. Training might fail evaluation steps.")
        val_ds = None

    # Test set (Use physical dir if exists, else error)
    if test_dir.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(test_dir, **loader_params)
    else:
        print("Warning: Test directory not found. Reporting will be limited.")
        test_ds = None

    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    if val_ds:
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    if test_ds:
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

if __name__ == "__main__":
    # Test loading
    DATA_DIR = os.path.join(os.path.dirname(__file__), "../dataset/train")
    try:
        train_ds, val_ds, classes = load_dataset(DATA_DIR)
        print(f"Successfully loaded dataset with {len(classes)} classes.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
