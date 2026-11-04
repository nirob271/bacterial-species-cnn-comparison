<div align="center">

# Efficient Bacterial Species Identification from Microscopic Images

### A Comparative CNN Benchmark on the DIBaS Dataset

**Saiful Islam Nirob · Minhaz Uddin Ahmed**

**Deep Learning | Computer Vision | Transfer Learning | Medical Image Analysis**

</div>

---

## 📌 Project Overview

This project presents a comprehensive benchmarking study of five ImageNet-pretrained Convolutional Neural Networks (CNNs) for bacterial species identification from microscopic images using the DIBaS dataset.

The primary objective is to investigate the trade-off between classification performance and computational efficiency across different CNN architectures under a unified experimental protocol.

Rather than introducing a new neural network architecture, this work focuses on a controlled comparative evaluation of existing architectures using consistent preprocessing, two-stage transfer learning, stratified five-fold cross-validation, statistical analysis, computational profiling, and model interpretability techniques.

The study evaluates not only classification accuracy but also macro-F1 score, model parameters, computational complexity, model size, GPU/CPU inference latency, and attention visualization.

## 🎯 Research Objectives

- Compare five ImageNet-pretrained CNN architectures for bacterial species classification.
- Investigate classification performance using stratified five-fold cross-validation.
- Implement a consistent two-stage transfer learning strategy.
- Evaluate model performance using multiple classification metrics.
- Perform statistical comparisons between model predictions.
- Analyze computational complexity and hardware-dependent inference latency.
- Investigate model interpretability using Grad-CAM, Grad-CAM++, and ScoreCAM.
- Examine within-class image similarity and potential limitations of dataset-level generalization.

## 🧠 Models Evaluated

Five CNN architectures are implemented and evaluated:

| Model | Architecture Type | Framework |
|---|---|---|
| MobileNetV3-Large | Lightweight CNN | PyTorch / Torchvision |
| EfficientNet-B0 | Efficient CNN | PyTorch / Torchvision |
| ResNet18 | Residual CNN | PyTorch / Torchvision |
| MobileNetV4-Conv-Small | Mobile-oriented CNN | PyTorch / timm |
| ConvNeXt-Tiny | Modern CNN | PyTorch / timm |

All five models use ImageNet-1K pretrained weights and are adapted for the 33-class classification task.

## 📂 Dataset Description

The experiments utilize the **DIBaS (Digital Image of Bacterial Species)** dataset.

| Dataset Characteristics | Description |
|---|---|
| Total raw images | 692 |
| Corrupted images removed | 3 |
| Final usable images | 689 |
| Number of classes | 33 |
| Original image resolution | 2048 × 1532 |
| Cached image resolution | 512-pixel long side |
| Model input resolution | 224 × 224 |

The dataset contains 32 bacterial taxa and *Candida albicans*.

### Dataset Preprocessing

The preprocessing pipeline includes:

1. Dataset downloading and integrity verification.
2. Identification and exclusion of corrupted TIFF images.
3. Image resizing while preserving aspect ratio.
4. JPEG conversion and caching.
5. Image normalization using ImageNet mean and standard deviation.
6. Training-time data augmentation.

Training augmentation includes random resized cropping, horizontal and vertical flipping, rotation, and color jitter.

Validation and test images are resized and center-cropped without augmentation.

**Note:** The dataset is not included in this repository. The notebook downloads the dataset from its original source.

## ⚙️ Methodology

The complete experimental pipeline consists of the following stages:

### 1. Dataset Audit and Preprocessing
- Download and inspect the DIBaS dataset.
- Identify corrupted or unreadable images.
- Generate a processed image cache.
- Verify class distribution and dataset integrity.

### 2. Stratified Five-Fold Cross-Validation

A stratified five-fold cross-validation strategy is implemented.

- Each image is used as a test sample exactly once.
- The remaining data are divided into training and validation subsets.
- Training, validation, and test splits are generated before augmentation.
- Data augmentation is applied exclusively to training images.
- Split disjointness is verified for every fold.

### 3. Two-Stage Transfer Learning

Each CNN follows a two-stage training strategy.

**Stage 1: Classification Head Training**
- Freeze backbone parameters.
- Train the classification head.
- Learning rate: 1e-3.
- Training duration: 5 epochs.

**Stage 2: Partial Fine-Tuning**
- Unfreeze selected deeper layers.
- Fine-tune the network.
- Learning rate: 1e-4.
- Maximum epochs: 30.
- Early stopping based on validation macro-F1.
- Patience: 5 epochs.

### 4. Model Evaluation

The following evaluation metrics are calculated:

- Accuracy
- Precision
- Recall
- Macro-F1 Score
- Weighted F1 Score
- Per-class classification metrics
- Confusion matrix and class-level error analysis

### 5. Statistical Analysis

To investigate differences between model predictions, the implementation includes:

- Fold-wise mean and standard deviation.
- Pooled out-of-fold (OOF) evaluation.
- 95% bootstrap confidence intervals.
- Exact McNemar tests for paired accuracy comparisons.
- Holm correction for multiple comparisons.
- Paired bootstrap confidence intervals for macro-F1 differences.

### 6. Computational Efficiency Analysis

The computational evaluation includes:

- Total trainable/model parameters.
- GMac computational complexity.
- Serialized model size.
- GPU inference latency.
- CPU inference latency.
- Single-thread CPU inference latency.

Inference latency is measured using batch size 1 with 224 × 224 input images.

### 7. Model Interpretability

Three visualization techniques are implemented:

- Grad-CAM
- Grad-CAM++
- ScoreCAM

Additionally, a deletion-based quantitative evaluation is performed to investigate the faithfulness of model attention.

### 8. Dataset Similarity and Leakage Audit

A frozen ImageNet-pretrained ResNet18 is used to extract image representations.

The notebook performs:

- Nearest-neighbour similarity analysis.
- Cross-split image similarity investigation.
- Frozen-feature 1-nearest-neighbour classification.
- Examination of potential within-class image similarity.

This analysis helps investigate the limitations of the dataset and the interpretation of cross-validation results.

---

## 📊 Experimental Results

### Classification Performance

The following table presents the five-fold cross-validation results.

| Model | Accuracy (%) | Macro-F1 (%) | Pooled OOF Macro-F1 (%) |
|---|---:|---:|---:|
| MobileNetV3-Large | 92.89 ± 3.56 | 92.38 ± 4.18 | 92.55 |
| EfficientNet-B0 | 90.86 ± 2.14 | 90.26 ± 2.12 | 90.51 |
| ResNet18 | **93.62 ± 5.69** | **93.41 ± 5.84** | **93.45** |
| MobileNetV4-Conv-Small | 84.34 ± 7.06 | 83.26 ± 7.36 | 83.85 |
| ConvNeXt-Tiny | 91.44 ± 0.95 | 90.64 ± 0.76 | 90.63 |

*Note: Accuracy and Macro-F1 are reported as mean ± standard deviation across five folds. Pooled OOF Macro-F1 is reported with a 95% bootstrap confidence interval in the manuscript.*

### Computational Performance

| Model | Parameters (M) | GMac | Model Size (MB) | GPU Latency (ms) | CPU Latency (ms) |
|---|---:|---:|---:|---:|---:|
| MobileNetV3-Large | 4.244 | 0.234 | 16.4 | 5.80 | 19.6 |
| EfficientNet-B0 | 4.050 | 0.414 | 15.7 | 7.73 | 33.4 |
| ResNet18 | 11.193 | 1.824 | 42.8 | 3.13 | 55.8 |
| MobileNetV4-Conv-Small | 2.535 | 0.185 | 9.9 | 4.69 | 13.4 |
| ConvNeXt-Tiny | 27.846 | 4.455 | 106.3 | 5.61 | 116.9 |

*Latency represents batch-size-1 inference measurements. GPU and CPU measurements were obtained using an NVIDIA Tesla T4 GPU and an Intel Xeon CPU environment in Google Colab.*

### Key Findings

- ResNet18 achieved the highest mean classification accuracy (93.62%) and macro-F1 score (93.41%).
- MobileNetV3-Large achieved 92.89% accuracy with only 4.244 million parameters.
- MobileNetV4-Conv-Small demonstrated the lowest computational complexity and fastest single-thread CPU inference.
- ResNet18 achieved the fastest measured GPU inference latency (3.13 ms).
- ConvNeXt-Tiny required substantially more computational resources without providing a corresponding classification-performance advantage.
- Statistical testing found no significant pairwise accuracy differences among the four stronger-performing models after Holm correction.
- MobileNetV4-Conv-Small performed significantly worse than the other four models under the shared training protocol.
- The frozen-feature 1-nearest-neighbour baseline achieved 88.25% accuracy, indicating substantial within-class image similarity.

These findings demonstrate that classification performance, model complexity, and inference latency do not necessarily improve together. Model selection depends on the intended hardware and computational constraints.

## 🛠️ Technologies and Libraries

The project uses the following technologies:

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| Torchvision | Pretrained CNN architectures |
| timm | Additional CNN architectures |
| NumPy | Numerical computation |
| Pandas | Dataset and result management |
| Scikit-learn | Cross-validation and evaluation |
| SciPy | Statistical analysis |
| Matplotlib | Data visualization |
| Grad-CAM | Model interpretability |
| THOP / PyTorch FLOP Counter | Computational profiling |
| Google Colab | Experimental environment |

## 📁 Repository Structure

```text
DIBaS-CNN-Benchmark/
│
├── README.md
│
├── DIBaS_CNN_Revision_5fold_CV.ipynb
│
├── results/
│   ├── figures/
│   ├── tables/
│   ├── runs/
│   ├── checkpoints/
│   ├── config.json
│   └── requirements_used.txt
│
└── paper/
    └── research_paper.pdf
