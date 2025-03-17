# CIFAR-10 Vision Transformer on AMD Instinct MI300X

> **Mixed Precision (AMP)** • **Up to 50 Epochs** • **RandAugment** • **\~84% Final Accuracy** • **Early Stop at 90%**

This repository showcases a Vision Transformer (ViT) trained on **CIFAR-10** with **PyTorch** using **Automatic Mixed Precision** on an **AMD Instinct MI300X GPU**. The script stops early if test accuracy reaches 90%, otherwise runs for 50 epochs. Final logs report ~**84.02%** best test accuracy in ~**13.18 minutes** of training.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Results](#key-results)
3. [Repository Structure](#repository-structure)
4. [Usage](#usage)
5. [Visualizations](#visualizations)
6. [Future Work](#future-work)
7. [License](#license)

---

## Overview

- **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  
  - 50k training images, 10k test images, 10 classes, each 32×32.  
- **Model**: Custom Vision Transformer  
  - Patch size = 4, 8 Transformer layers, 384 embedding dimension, 6 attention heads.  
  - **RandAugment** for stronger data augmentation.  
- **Mixed Precision**: PyTorch’s `autocast(enabled=True)` for half-precision training on MI300X.  
- **Training**: Batch size = 128, up to 50 epochs, early-stop at 90% accuracy.  
- **Inference**: Benchmarks latencies and throughput for batch sizes [1,4,16,64,128].  

### Why This Project?

1. Demonstrates **AMP** (Automatic Mixed Precision) for **faster training** on AMD GPUs.  
2. Showcases how a **Vision Transformer** can achieve **\~84%** on CIFAR-10 in **\~13 minutes**.  
3. Provides reference code and plots for **inference scaling** (throughput vs. batch size).

---

## Key Results

1. **Final Accuracy**: ~**84.02%**  
2. **Training Time**: ~**13.18 minutes** (on MI300X)  
3. **Throughput** (inference, from `inference_stats.json`):
   ```json
   {
     "1":   { "inference_time_ms": 3.47, "images_per_sec": 288.09 },
     "4":   { "inference_time_ms": 3.74, "images_per_sec": 1068.95 },
     "16":  { "inference_time_ms": 4.33, "images_per_sec": 3697.96 },
     "64":  { "inference_time_ms": 5.29, "images_per_sec": 12087.71 },
     "128": { "inference_time_ms": 6.25, "images_per_sec": 20482.77 }
   }
   ```

---

## Repository Structure

```plaintext
.
├── train_mi300x.py              # Main training script
├── results/
│   ├── sample_predictions.png   # 5 random test images (True vs. Pred)
│   ├── training_metrics.png     # Loss & Accuracy curves
│   ├── inference_stats.json     # Latency & throughput for batch sizes
│   └── mi300x_throughput_scaling.png (optional scaling plot)
├── checkpoints/
│   └── vit_final_mi300x.pth     # Final model checkpoint
├── analysis_mi300x.py           # (Optional) script to plot throughput scaling
└── README.md                    # This file

---

## Usage

1. **Environment Setup**  
   - Python ≥3.9  
   - PyTorch with ROCm support (for AMD GPU)  
   - `torchvision`, `tqdm`, `matplotlib`, `numpy`, etc.

2. **Install** dependencies (example):
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.4.2
   pip install tqdm matplotlib pandas
   ```

3. **Train**:
   ```bash
   python train_mi300x.py
   ```
   - Automatically downloads CIFAR-10
   - Trains up to 50 epochs, early stopping if accuracy ≥90%
   - Outputs logs & images to `results/`  
   - Checkpoint saved at `checkpoints/vit_final_mi300x.pth`

4. **Analyze**:
   - **Training logs** show final ~84.02% accuracy at epoch ~49/50.  
   - **`results/training_metrics.png`** for epoch-by-epoch curves.  
   - **`results/sample_predictions.png`** for random test images.

5. **Inference**:
   - Run automatically after training in the same script.  
   - See throughput & latency in `results/inference_stats.json`.  
   - Optional scaling plot in `mi300x_throughput_scaling.png` (if you have a separate analysis script).

---

## Visualizations

### 1) Sample Predictions!

[sample_predictions](https://github.com/user-attachments/assets/c55d83ac-34f6-4096-85b9-5b80d1e3cbde)



Shows random test images with “True vs. Pred” labels. Titles are clearly spaced to avoid overlap.

### 2) Training Metrics

![training_metrics](https://github.com/user-attachments/assets/1af43994-e133-4e9e-b7fa-b9b9bf1eea9e)

- **Left**: Train vs. Test Loss  
- **Right**: Train vs. Test Accuracy  

### 3) Throughput Scaling

If you run the optional analysis script, you get `mi300x_throughput_scaling.png`:

![mi300x_throughput_scaling](https://github.com/user-attachments/assets/0ea6f654-5788-4eed-9efd-39b28badcf2f)


- **Left**: Actual vs. Ideal throughput (batch_size=1 as baseline)  
- **Right**: Scaling efficiency in %

---

## Future Work

- **Longer Training**: Increase beyond 50 epochs for potential 90%+ accuracy.  
- **Bigger ViT**: Raise embedding dim or layers.  
- **Advanced Augmentations**: E.g. Mixup, CutMix, or additional RandAugment ops.  
- **Bigger Dataset**: Try CIFAR-100 or ImageNet for more comprehensive results.

---

## License

This project is released under the [MIT License](LICENSE).  
&copy; 2025 YourName. All rights reserved.  
```
