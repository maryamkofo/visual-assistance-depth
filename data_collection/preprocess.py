import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

rgb_dir = "dataset/rgb"
depth_dir = "dataset/depth"

print("Scanning dataset...")

valid_samples = []

for fname in sorted(os.listdir(rgb_dir)):
    if not fname.endswith(".png"):
        continue

    idx = fname.replace(".png", "")
    depth_path = os.path.join(depth_dir, f"{idx}.npy")

    # Skip if depth file doesn't exist
    if not os.path.exists(depth_path):
        continue

    depth = np.load(depth_path)

    # Check how many pixels have valid depth (non-zero)
    valid_pixels = np.sum(depth > 0) / depth.size

    if valid_pixels > 0.7:  # keep frames with more than 70% valid depth
        valid_samples.append(idx)

print(f"Total valid frames: {len(valid_samples)}")

# Split into train and validation sets (85% train, 15% val)
train, val = train_test_split(valid_samples, test_size=0.15, random_state=42)

print(f"Train: {len(train)} frames")
print(f"Val:   {len(val)} frames")

# Save the splits to text files
os.makedirs("dataset/splits", exist_ok=True)

with open("dataset/splits/train.txt", "w") as f:
    f.write("\n".join(train))

with open("dataset/splits/val.txt", "w") as f:
    f.write("\n".join(val))

print("Saved splits to dataset/splits/train.txt and dataset/splits/val.txt")
