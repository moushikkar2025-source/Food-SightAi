
import os
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def verify_image(file_path):
    """
    Verify if an image file is valid and not corrupt.
    Returns True if valid, False otherwise.
    """
    try:
        if os.path.getsize(file_path) == 0:
            return False
        
        with Image.open(file_path) as img:
            img.verify() # Verify file integrity
        
        # Re-open to check if it can be loaded and converted
        with Image.open(file_path) as img:
             img.convert('RGB')
        
        return True
    except (IOError, SyntaxError, OSError):
        return False

def prepare_dataset(source_dir, target_dir, cap_limit=600, ratio=(0.7, 0.2, 0.1)):
    """
    Clean, cap, split, and organize dataset.
    
    Args:
        source_dir (str): Path to raw dataset containing class folders.
        target_dir (str): Path to destination dataset directory.
        cap_limit (int): Maximum images per class.
        ratio (tuple): Train, validation, test split ratio.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {source_path}")
        return

    print("=" * 60)
    print("DATASET CAPPING & SPLITTING PIPELINE")
    print("=" * 60)
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print(f"Cap Limit: {cap_limit} images per class")
    print(f"Split Ratio: Train={ratio[0]}, Val={ratio[1]}, Test={ratio[2]}")
    print("-" * 60)

    # 1. Scan for classes
    classes = sorted([d for d in source_path.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes in source directory.")
    
    # 2. Process each class
    stats = {
        'total_scanned': 0,
        'valid_images': 0,
        'capped_images': 0,
        'corrupt_files': 0,
        'train': 0,
        'val': 0,
        'test': 0
    }
    
    for class_dir in tqdm(classes, desc="Processing Classes"):
        class_name = class_dir.name
        
        # Get all image files with standard extensions
        potential_images = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_extensions:
                potential_images.append(f)
        
        stats['total_scanned'] += len(potential_images)
        
        # Filter valid images until we reach the cap
        # We shuffle first to ensure we pick a random subset if it's over the cap
        random.shuffle(potential_images)
        
        valid_capped_images = []
        for img_path in potential_images:
            if verify_image(img_path):
                valid_capped_images.append(img_path)
                if len(valid_capped_images) >= cap_limit:
                    break
            else:
                stats['corrupt_files'] += 1
        
        n_capped = len(valid_capped_images)
        stats['valid_images'] += n_capped # Note: this is actually 'valid and picked'
        stats['capped_images'] += n_capped
        
        # Split the capped subset
        random.shuffle(valid_capped_images)
        n_train = int(n_capped * ratio[0])
        n_val = int(n_capped * ratio[1])
        
        splits = {
            'train': valid_capped_images[:n_train],
            'validation': valid_capped_images[n_train:n_train+n_val],
            'test': valid_capped_images[n_train+n_val:]
        }
        
        # Copy to target splits
        for split_name, split_imgs in splits.items():
            split_dest = target_path / split_name / class_name
            split_dest.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_imgs:
                # Use class_name in filename to avoid any potential collisions
                dest_file = split_dest / img_path.name
                try:
                    shutil.copy2(img_path, dest_file)
                    if split_name == 'validation':
                        stats['val'] += 1
                    else:
                        stats[split_name] += 1
                except Exception as e:
                    print(f"[Error] Failed to copy {img_path.name}: {e}")

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total Files Scanned:      {stats['total_scanned']}")
    print(f"Corrupt Files Skipped:    {stats['corrupt_files']}")
    print(f"Total Capped Images:      {stats['capped_images']}")
    print("-" * 30)
    print(f"Final Train Images:       {stats['train']}")
    print(f"Final Validation Images:  {stats['val']}")
    print(f"Final Test Images:        {stats['test']}")
    print("=" * 60)
    print(f"\nFinal dataset ready at: {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cap, clean, and split dataset.")
    parser.add_argument("--source", required=True, help="Path to raw merged dataset")
    parser.add_argument("--dest", required=True, help="Path to destination dataset folder")
    parser.add_argument("--cap", type=int, default=600, help="Max images per class (default: 600)")
    
    args = parser.parse_args()
    
    # Run the process
    prepare_dataset(args.source, args.dest, cap_limit=args.cap)
