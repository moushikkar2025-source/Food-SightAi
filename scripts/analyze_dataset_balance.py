import os
import matplotlib.pyplot as plt

DATASET_DIR = r"c:\Users\TANMAY\OneDrive\Desktop\PROJECT\MULTIDISPLINARY PROJECT\Review_4\FoodSight-Ai-App\dataset"
MIN_IMAGES_REQUIRED = 150  # Threshold for "needs more photos"

def analyze_dataset():
    print(f"Analyzing dataset at: {DATASET_DIR}")
    
    class_counts = {}
    total_images = 0
    
    # Count images per class
    train_dir = os.path.join(DATASET_DIR, "train")
    if not os.path.exists(train_dir):
        print("Error: 'train' directory not found!")
        return

    classes = sorted(os.listdir(train_dir))
    
    for class_name in classes:
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
            class_counts[class_name] = count
            total_images += count
            
    # Sort by count (ascending)
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    
    print("\n" + "="*60)
    print(f"DATASET ANALYSIS")
    print("="*60)
    print(f"Total Classes: {len(classes)}")
    print(f"Total Images: {total_images}")
    print(f"Average Images/Class: {total_images / len(classes):.1f}")
    print("-" * 60)
    
    print(f"\nCLASSES NEEDING MORE PHOTOS (<{MIN_IMAGES_REQUIRED}):")
    print("-" * 60)
    print(f"{'Class Name':<30} | {'Count':<10} | {'Deficit':<10}")
    print("-" * 60)
    
    needs_more = []
    
    for class_name, count in sorted_counts:
        if count < MIN_IMAGES_REQUIRED:
            deficit = MIN_IMAGES_REQUIRED - count
            print(f"{class_name:<30} | {count:<10} | +{deficit:<10}")
            needs_more.append((class_name, deficit))
            
    if not needs_more:
        print("Good news! All classes have sufficient images.")
    else:
        print("-" * 60)
        print(f"Total classes needing augmentation: {len(needs_more)}")

    # Generate a simple bar chart
    plt.figure(figsize=(20, 10))
    names = [x[0] for x in sorted_counts]
    values = [x[1] for x in sorted_counts]
    
    colors = ['red' if x < MIN_IMAGES_REQUIRED else 'green' for x in values]
    
    plt.bar(names, values, color=colors)
    plt.xticks(rotation=90, fontsize=8)
    plt.axhline(y=MIN_IMAGES_REQUIRED, color='r', linestyle='-', label=f'Min Required ({MIN_IMAGES_REQUIRED})')
    plt.title("Images per Class (Red = Needs More)")
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    print("\n[INFO] Saved distribution chart to 'dataset_distribution.png'")

if __name__ == "__main__":
    analyze_dataset()
