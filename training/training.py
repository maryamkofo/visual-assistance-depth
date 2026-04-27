import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DepthDataset(Dataset):
    def __init__(self, split, processor, splits_dir="dataset/splits", 
                 rgb_dir="dataset/rgb", depth_dir="dataset/depth"):
        """
        split: 'train' or 'val'
        processor: Hugging Face image processor for Depth-Anything-V2
        """
        self.processor = processor
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir

        # Load the list of frame IDs for this split
        split_file = os.path.join(splits_dir, f"{split}.txt")
        with open(split_file, "r") as f:
            self.sample_ids = [line.strip() for line in f.readlines()]

        print(f"Loaded {len(self.sample_ids)} samples for {split}")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]

        # Load RGB image
        rgb_path = os.path.join(self.rgb_dir, f"{sid}.png")
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load depth map (in millimeters) and convert to meters
        depth_path = os.path.join(self.depth_dir, f"{sid}.npy")
        depth = np.load(depth_path).astype(np.float32)
        depth = depth / 1000.0  # convert mm to meters

        # Process RGB image for the model
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs, torch.tensor(depth)
