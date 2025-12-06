"""
ImageNet-50 Dataset Visualization
==================================
Loads from HuggingFace to get proper class labels.

Usage: python imagenet50_visualize_hf.py

Requirements: pip install datasets pillow matplotlib
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset

# ============================================================
# Configuration
# ============================================================
IMAGES_PER_CLASS = 4
NUM_CLASSES_COMPACT = 12  # For the compact report figure
OUTPUT_DIR = "."

# ============================================================
# Main Functions
# ============================================================

def load_imagenet50():
    """
    Load ImageNet-50 from HuggingFace with proper labels.
    """
    print("Loading ImageNet-50 dataset from HuggingFace...")
    print("(This may take a moment on first run)\n")
    
    dataset = load_dataset("Elriggs/imagenet-50-subset", split="train")
    
    print(f"✓ Loaded {len(dataset)} images")
    
    # Get class names
    if hasattr(dataset.features['label'], 'names'):
        class_names = dataset.features['label'].names
        print(f"✓ Found {len(class_names)} class names")
    else:
        class_names = None
        print("⚠ Class names not found in features")
    
    return dataset, class_names


def organize_by_class(dataset, class_names, images_per_class=4):
    """
    Group images by their class label.
    Returns: {class_id: {"name": str, "images": [PIL.Image]}}
    """
    print("\nOrganizing images by class...")
    
    class_data = {}
    
    for sample in dataset:
        label = sample['label']
        
        if label not in class_data:
            # Get class name
            if class_names and label < len(class_names):
                name = class_names[label]
            elif 'class_name' in sample:
                name = sample['class_name']
            else:
                name = f"Class {label}"
            
            class_data[label] = {
                "name": name,
                "images": []
            }
        
        # Add image if we need more
        if len(class_data[label]["images"]) < images_per_class:
            img = sample['image']
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert('RGB')
            class_data[label]["images"].append(img)
        
        # Check if we have enough for all classes
        if all(len(d["images"]) >= images_per_class for d in class_data.values()):
            if len(class_data) >= 50:  # We expect ~50 classes
                break
    
    print(f"✓ Organized {len(class_data)} classes")
    return class_data


def create_compact_grid(class_data, num_classes=12, output_path="dataset_grid_12classes.png"):
    """
    Create a compact grid for the report (first N classes).
    """
    print(f"\nCreating compact grid ({num_classes} classes)...")
    
    sorted_labels = sorted(class_data.keys())[:num_classes]
    
    fig, axes = plt.subplots(num_classes, IMAGES_PER_CLASS, 
                             figsize=(10, num_classes * 2.2))
    
    fig.suptitle('ImageNet-50 Dataset Samples\n(4 images per class)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    for row_idx, label in enumerate(sorted_labels):
        data = class_data[label]
        class_name = data["name"]
        images = data["images"]
        
        for col_idx in range(IMAGES_PER_CLASS):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(images):
                img = images[col_idx]
                # Resize for display
                img = img.resize((150, 150))
                ax.imshow(img)
            
            ax.axis('off')
        
        # Add class name on the left
        axes[row_idx, 0].text(
            -0.1, 0.5, class_name[:25],
            transform=axes[row_idx, 0].transAxes,
            fontsize=9, fontweight='bold',
            va='center', ha='right'
        )
    
    plt.tight_layout(rect=[0.15, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_full_grid(class_data, output_path="dataset_grid_all50.png"):
    """
    Create a comprehensive grid with all classes.
    Layout: 10 rows x 5 groups (each group = 4 images for one class)
    """
    print("\nCreating full grid (all classes)...")
    
    num_classes = len(class_data)
    sorted_labels = sorted(class_data.keys())
    
    # Layout: 10 rows, 5 class-groups per row, 4 images per class
    rows = 10
    groups_per_row = 5
    cols = groups_per_row * IMAGES_PER_CLASS  # 20 columns
    
    fig, axes = plt.subplots(rows, cols, figsize=(24, 18))
    fig.suptitle('ImageNet-50 Dataset - All Classes\n(4 sample images per class)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for class_idx, label in enumerate(sorted_labels):
        if class_idx >= rows * groups_per_row:
            break
            
        data = class_data[label]
        class_name = data["name"]
        images = data["images"]
        
        row = class_idx % rows
        group = class_idx // rows
        
        for img_idx in range(IMAGES_PER_CLASS):
            col = group * IMAGES_PER_CLASS + img_idx
            ax = axes[row, col]
            
            if img_idx < len(images):
                img = images[img_idx].resize((112, 112))
                ax.imshow(img)
            
            ax.axis('off')
            
            # Add class name above first image
            if img_idx == 0:
                display_name = class_name[:18] + "..." if len(class_name) > 18 else class_name
                ax.set_title(display_name, fontsize=6, fontweight='bold', pad=2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def create_category_grid(class_data, output_path="dataset_grid_categories.png"):
    """
    Create a grid organized by semantic categories.
    """
    print("\nCreating category-organized grid...")
    
    # Define categories based on ImageNet-50 class order
    # Classes 0-6: Fish, 7-24: Birds, 25-32: Amphibians, 33-49: Reptiles
    categories = {
        "Fish": list(range(0, 7)),
        "Birds": list(range(7, 25)),
        "Amphibians": list(range(25, 33)),
        "Reptiles": list(range(33, 50))
    }
    
    # Select 3 classes from each category
    fig, axes = plt.subplots(4, 3 * IMAGES_PER_CLASS, figsize=(18, 10))
    fig.suptitle('ImageNet-50 Dataset by Category\n(3 example classes per category)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    for cat_idx, (cat_name, label_range) in enumerate(categories.items()):
        # Get first 3 available classes in this category
        available = [l for l in label_range if l in class_data][:3]
        
        for class_offset, label in enumerate(available):
            data = class_data[label]
            class_name = data["name"]
            images = data["images"]
            
            for img_idx in range(IMAGES_PER_CLASS):
                col = class_offset * IMAGES_PER_CLASS + img_idx
                ax = axes[cat_idx, col]
                
                if img_idx < len(images):
                    img = images[img_idx].resize((120, 120))
                    ax.imshow(img)
                
                ax.axis('off')
                
                if img_idx == 0:
                    ax.set_title(class_name[:15], fontsize=7, pad=2)
        
        # Add category label
        axes[cat_idx, 0].text(
            -0.15, 0.5, cat_name,
            transform=axes[cat_idx, 0].transAxes,
            fontsize=11, fontweight='bold',
            va='center', ha='right',
            color='darkblue'
        )
    
    plt.tight_layout(rect=[0.1, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def print_dataset_info(class_data):
    """
    Print dataset summary for the report.
    """
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal classes: {len(class_data)}")
    print(f"\nClass list:")
    
    for i, (label, data) in enumerate(sorted(class_data.items())):
        print(f"  {i+1:2d}. {data['name']}")
    
    print("\n" + "=" * 60)


def main():
    print("=" * 60)
    print("ImageNet-50 Dataset Visualization")
    print("=" * 60)
    
    # Load dataset
    dataset, class_names = load_imagenet50()
    
    # Organize by class
    class_data = organize_by_class(dataset, class_names, IMAGES_PER_CLASS)
    
    # Print info
    print_dataset_info(class_data)
    
    # Create visualizations
    create_compact_grid(class_data, num_classes=12, 
                       output_path="dataset_grid_12classes.png")
    
    create_full_grid(class_data, 
                    output_path="dataset_grid_all50.png")
    
    create_category_grid(class_data,
                        output_path="dataset_grid_categories.png")
    
    print("\n" + "=" * 60)
    print("Done! Generated files:")
    print("  1. dataset_grid_12classes.png  (compact, for report)")
    print("  2. dataset_grid_all50.png      (all 50 classes)")
    print("  3. dataset_grid_categories.png (organized by category)")
    print("=" * 60)


if __name__ == "__main__":
    main()