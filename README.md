
# Vision Transformer on AMD Instinct MI300X

> **Mixed Precision (AMP)** • **Up to 50 Epochs** • **RandAugment** • **\~84% Final Accuracy** • **Stop at 90%**  

This repository features a **Vision Transformer** (ViT) for the **CIFAR-10** dataset, trained in **mixed precision** (using AMP) on an **AMD Instinct MI300X** GPU.  
All code is provided as a **Jupyter Notebook**:  
**`CIFAR-10 Vision Transformer with Mixed Precision on AMD Instinct MI300X.ipynb`**.

---

## Table of Contents

1. [Overview](#overview)  
2. [Key Results](#key-results)  
3. [Setup & Usage](#setup--usage)  
4. [Notebook Sections](#notebook-sections)  
5. [Outputs & Visualizations](#outputs--visualizations)  
6. [Monitoring GPU Usage](#monitoring-gpu-usage)  
7. [Future Work](#future-work)  
8. [License](#license)

---

## Overview

- **Dataset**: CIFAR-10 (32×32 images, 10 classes, 50k training + 10k test).  
- **Model**: Vision Transformer with patch size=4, 8 layers, embed_dim=384, 6 heads, **RandAugment** for data augmentation.  
- **Precision**: Standard AMP (`autocast(enabled=True)`) on AMD Instinct MI300X.  
- **Training**: 50 epochs (or stop early at 90% test accuracy), batch size=128.  
- **Inference**: Benchmarks throughput & latency at batch sizes [1,4,16,64,128], logs to `inference_stats.json`.  

## Key Results

1. **Final Accuracy**: ~84.02%  
2. **Training Time**: ~13.18 minutes  
3. **Sample Predictions**: A few random test images with “True vs Pred” labels (see `sample_predictions.png`).  
4. **Inference**:  
   - Latency & throughput for each batch size in `inference_stats.json`.  
   - Example: ~20K images/sec at batch size=128.

---

## Setup & Usage

### 1. Install Dependencies

- **Python** ≥ 3.9  
- **PyTorch** for ROCm 6.1 (or whichever ROCm version you use)  
- `torchvision`, `tqdm`, `matplotlib`, `numpy`, etc.

```bash
# Example: installing PyTorch for ROCm 6.1
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm6.1
pip install tqdm matplotlib pandas
```

### 2. Launch Jupyter & Open Notebook

1. Start Jupyter (Lab or Notebook):
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
2. Open the **`CIFAR-10 Vision Transformer with Mixed Precision on AMD Instinct MI300X.ipynb`** file.

### 3. Run the Notebook

- The notebook will:
  - **Download** CIFAR-10 automatically.
  - **Train** up to 50 epochs (stopping early if accuracy ≥ 90%).  
  - **Plot** training curves & sample predictions.  
  - **Benchmark** inference and save stats in `results/inference_stats.json`.  

---

## Notebook Sections

1. **Hyperparameters & Configuration**: Edit batch size, epochs, model depth, etc.  
2. **Dataloaders** (CIFAR-10 + RandAugment).  
3. **Vision Transformer Blocks** (PatchEmbedding, MLP, MultiHeadAttention, etc.).  
4. **Training Loop** (AMP, gradient scaling, early stop).  
5. **Evaluation** (test accuracy, sample predictions).  
6. **Inference Benchmark** (batch sizes, throughput, latencies).  
7. **Plots** (loss/accuracy curves, sample predictions, etc.).

---

## Outputs & Visualizations

| **File**                             | **Description**                                                            |
|--------------------------------------|----------------------------------------------------------------------------|
| **`results/training_metrics.png`**   | Two-panel plot: (Loss vs Epoch) and (Accuracy vs Epoch)                    |
| **`results/sample_predictions.png`** | Random test images showing “True” vs “Pred” labels                         |
| **`results/inference_stats.json`**   | Inference throughput (images/sec) & latency (ms) for batch sizes [1..128]  |
| **`checkpoints/vit_final_mi300x.pth`** | Final trained model weights in PyTorch format                              |
| **`analysis/mi300x_throughput_scaling.png`** | Throughput vs. batch size plot, if you run the analysis script |

### Sample Predictions

![sample_predictions](https://github.com/user-attachments/assets/de94dc86-168d-43f9-a109-cf4095c37bda)

**Description**: Shows random test images with “True vs. Pred” labels. Titles are clearly spaced to avoid overlap.

### Training Metrics

![training_metrics](https://github.com/user-attachments/assets/2d359bf8-6298-430a-b6e8-c57037ce42a7)

- **Left**: Train vs. Test Loss  
- **Right**: Train vs. Test Accuracy  

### Throughput Scaling

If you run the analysis script, you get `mi300x_throughput_scaling.png`:

![mi300x_throughput_scaling](https://github.com/user-attachments/assets/fb77902c-a830-4c81-bf8f-c0feb2bf5fdd)

- **Left**: Actual vs. Ideal throughput (batch_size=1 as baseline)  
- **Right**: Scaling efficiency in %

**Sample**:
```
[Epoch 49/50] Train Acc=82.10% | Test Acc=84.02%
[Epoch 50/50] Train Acc=82.03% | Test Acc=83.93%
[MAIN] Training completed in 13.18 minutes. Best test acc=84.02%
[INFO] Inference stats saved: results/inference_stats.json
```

---

## Monitoring GPU Usage

While training, you can also track GPU usage in a separate terminal using **`rocm-smi`** every 2 seconds:

```bash
watch -n 1 rocm-smi --showtemp --showuse --showmeminfo=vram
```

![image](https://github.com/user-attachments/assets/1c1f63dd-2726-40d2-a53f-1d50c560c729)


This command repeatedly prints GPU stats—like utilization, memory usage, and temperature—so you can confirm that training is running smoothly. Feel free to include a screenshot in the notebook to document GPU load or temperature if you like.

---

## Future Work

- **More Epochs**: Possibly 100+ for higher accuracy.  
- **Larger ViT**: Increase embedding dimension or layers.  
- **Advanced Augment**: Mixup, CutMix, or deeper RandAugment settings.  
- **Bigger Dataset**: Test on CIFAR-100 or ImageNet for deeper benchmarks.

---

## License

© 2025 Sai Dhanush Kolla. All rights reserved.
```
