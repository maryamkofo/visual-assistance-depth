import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from dataset import DepthDataset

# CONFIGURATION
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-4
SAVE_PATH = "models/finetuned-depth-model"

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load processor and model 
print("Loading model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)
model = model.to(device)

# Freeze encoder, only train decoder
# speeds up training and prevents overfitting on a small dataset
for name, param in model.named_parameters():
    if "neck" in name or "head" in name:
        param.requires_grad = True   # train decoder
    else:
        param.requires_grad = False  # freeze encoder

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}")

# --- Load datasets ---
train_dataset = DepthDataset(split="train", processor=processor)
val_dataset = DepthDataset(split="val", processor=processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# OPTIMIZER
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# TRAINING LOOP
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for batch_idx, (inputs, depth_gt) in enumerate(train_loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        depth_gt = depth_gt.to(device)

        # Forward pass
        outputs = model(**inputs)
        pred = outputs.predicted_depth  # (B, H, W)

        # Resize ground truth to match prediction size
        depth_gt_resized = nn.functional.interpolate(
            depth_gt.unsqueeze(1),
            size=pred.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        # Only compute loss on valid pixels (ignore RealSense holes)
        mask = depth_gt_resized > 0
        loss = nn.functional.l1_loss(pred[mask], depth_gt_resized[mask])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # VALIDATION
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, depth_gt in val_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            depth_gt = depth_gt.to(device)

            outputs = model(**inputs)
            pred = outputs.predicted_depth

            depth_gt_resized = nn.functional.interpolate(
                depth_gt.unsqueeze(1),
                size=pred.shape[-2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(1)

            mask = depth_gt_resized > 0
            loss = nn.functional.l1_loss(pred[mask], depth_gt_resized[mask])
            val_loss += loss.item()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} complete | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

# Save fine-tuned model 
import os
os.makedirs(SAVE_PATH, exist_ok=True)
model.save_pretrained(SAVE_PATH)
processor.save_pretrained(SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")
