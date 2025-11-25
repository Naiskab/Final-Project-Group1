from datasets import load_dataset
import os
import csv
from PIL import Image

# Load dataset
dataset = load_dataset("Elriggs/imagenet-50-subset")

# Create output directory
output_dir = "imagenet_50"
os.makedirs(output_dir, exist_ok=True)

# Create CSV file
csv_path = os.path.join(output_dir, "labels.csv")
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)

# Write CSV header
writer.writerow(["image_id", "filename", "label"])

image_id = 0  

# Save images + record CSV rows
for split in ["train", "validation"]:
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    for item in dataset[split]:
        img = item["image"]
        label = item["label"]

        # Create filename
        filename = f"{split}_{image_id}.jpg"
        file_path = os.path.join(split_dir, filename)

        # Save the image
        img.save(file_path)

        # Write metadata to CSV
        writer.writerow([image_id, filename, label])

        # Increment image ID
        image_id += 1

csv_file.close()

print("Images and CSV saved at:", output_dir)
