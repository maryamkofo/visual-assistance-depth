# Real-time Visual Assistance Through Depth Estimation

A real-time assistive navigation system for visually impaired individuals using monocular depth estimation deployed on an embedded edge device (NVIDIA Jetson).

> CS [Course Number] Final Project — University of Texas at Dallas  
> Team: Maryam Sulaiman, Giovani Jonenson, Roslyn Collings, Mitchell Tuan Vu

---

## Overview

Traditional mobility aids like walking canes cannot detect elevated hazards such as overhanging branches, open doors, or signs. This system uses a single RGB camera mounted at chest level to estimate per-pixel depth in real time, detect nearby obstacles, and alert the user via audio or vibration feedback.

---

## Repository Structure

```
visual-assistance-depth/
├── data_collection/
│   ├── record_realsense.py       # Capture paired RGB + depth frames using Intel RealSense
│   └── preprocess.py             # Filter invalid frames, train/val split
│
├── training/
│   ├── dataset.py                # PyTorch Dataset class for RGB + depth pairs
│   ├── finetune.py               # Fine-tune Depth-Anything-V2 on collected data
│   └── evaluate.py               # Compute AbsRel, RMSE, delta metrics
│
├── pipeline/
│   └── main.py                   # Full real-time pipeline: webcam → depth → alert
│
├── models/
│   └── finetuned-depth-model/    # Saved fine-tuned model weights (git-ignored if large)
│
├── notebooks/
│   └── results_visualization.ipynb   # Visualize depth maps and evaluation results
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## System Pipeline

```
Webcam (RGB frames)
        ↓
Depth-Anything-V2 (monocular depth estimation)
        ↓
Depth Map (NumPy array, per-pixel distances)
        ↓
Velocity Map (frame-to-frame depth difference)
        ↓
Obstacle Detection (distance threshold + velocity threshold)
        ↓
Alert Generation (audio warning)
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
- `torch`, `torchvision`
- `transformers` (Hugging Face)
- `pyrealsense2`
- `opencv-python`
- `numpy`
- `scikit-learn`

### Running the Real-time Pipeline

```bash
python pipeline/main.py
```

### Collecting Your Own Data (Intel RealSense)

```bash
python data_collection/record_realsense.py
```

Frames are saved to `dataset/rgb/` and `dataset/depth/`. This folder is excluded from Git — share via Google Drive.

### Fine-tuning

```bash
python training/finetune.py
```

Trains for 5 epochs by default, freezing the encoder and fine-tuning the decoder head only. Model is saved to `models/finetuned-depth-model/`.

### Evaluation

```bash
python training/evaluate.py
```

Reports AbsRel, RMSE, and δ<1.25 on the validation set for both the baseline and fine-tuned model.

---

## Dataset

The dataset was collected using an Intel RealSense camera across three environments:
- Indoor hallways and corridors
- Rooms with furniture
- Outdoor sidewalks

Each sample consists of a paired RGB image and aligned depth map (in millimeters). The dataset is **not** included in this repository due to size. Team members can access it via the shared Google Drive folder.

---

## Results

| Model | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ |
|---|---|---|---|
| Depth-Anything-V2 (baseline) | TBD | TBD | TBD |
| Fine-tuned (ours) | TBD | TBD | TBD |

*Results will be updated after final evaluation.*

---

## References

1. CDC, "Fast Facts: Vision Loss," May 2024.
2. Yang et al., "Depth Anything V2," NeurIPS 2024.
3. Hugging Face Transformers — Depth Anything V2.
4. Ultralytics, "How to Use Depth Anything V2," YouTube, 2025.
