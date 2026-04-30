# Real-time Visual Assistance Through Depth Estimation

A real-time assistive navigation system for visually impaired individuals using monocular depth estimation deployed on an embedded edge device (NVIDIA Jetson).

> CS 4390 Final Project 
> Team: Maryam Sulaiman, Giovani Jonenson, Roslyn Collings, Mitchell Vu

---

## Overview

Traditional mobility aids like walking canes cannot detect elevated hazards such as overhanging branches, open doors, or signs. This system uses a single RGB camera mounted at chest level to estimate per-pixel depth in real time, detect nearby obstacles, and alert the user via audio or vibration feedback.

---

## Repository Structure

```
visual-assistance-depth/

├── training/
│   └── evaluate.py               # Compute AbsRel, RMSE, delta metrics
│
├── pipeline/
│   └── main.py                   # Full real-time pipeline: webcam → depth → alert
│
├── models/
│   └── finetuned-depth-model/    # Saved fine-tuned model weights
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



### Evaluation

```bash
python training/training-evaluation.py
```

Reports AbsRel, RMSE, and δ<1.25 on the validation set for both the baseline and fine-tuned model.

---

## Dataset

We used the pre-labled NYU Depth Dataset V2

---

## Results

| Model | AbsRel ↓ | RMSE ↓ | δ<1.25 ↑ |
|---|---|---|---|
| Depth-Anything-V2-S | .053 | - | .973 |
| NYUmodel (ours) | 0.1002 | 0.3934 |0.8988 |

*Results will be updated after final evaluation.*

---

## References

1. CDC, "Fast Facts: Vision Loss," May 2024.
2. Yang et al., "Depth Anything V2," NeurIPS 2024.
3. Hugging Face Transformers — Depth Anything V2.
4. Ultralytics, "How to Use Depth Anything V2," YouTube, 2025.
5. Kaggle, "Depth Anything V2 - Metric fine-tunning on NYU", 2023. 
