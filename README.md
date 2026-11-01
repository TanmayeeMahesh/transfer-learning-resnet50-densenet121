# Transfer Learning: ResNet-50 vs. DenseNet-121 on Caltech-101

A comparative study of two pretrained CNN architectures, ResNet-50 and DenseNet-121, using standard and adaptive fine-tuning strategies for image classification on the Caltech-101 dataset.

## Overview

Transfer learning allows deep learning models pretrained on large datasets such as ImageNet to be adapted to smaller target datasets.

This project investigates how different fine-tuning strategies affect classification performance, generalisation, training time, and parameter efficiency.

Two architectures are evaluated:

- **ResNet-50:** Uses residual connections to improve gradient flow.
- **DenseNet-121:** Uses dense connectivity to encourage feature reuse across layers.

Two fine-tuning approaches are compared:

1. Standard Fine-tuning — selectively unfreezing higher-level layer groups.
2. Adaptive Fine-tuning — selecting important convolutional filters based on their activation magnitudes and restricting gradient updates.

The experiments investigate whether filter-level importance ranking can provide a more parameter-efficient alternative to conventional layer-based fine-tuning.

## Objectives

- Compare ResNet-50 and DenseNet-121 on a small, visually diverse dataset.
- Evaluate different backbone unfreezing configurations.
- Implement activation-based adaptive filter selection.
- Compare last-block and global filter-importance strategies.
- Analyse accuracy, macro F1, training time, and trainable parameters.
- Study overfitting and generalisation under different fine-tuning configurations.

## Dataset

**Caltech-101**

The Caltech-101 dataset contains 9,144 RGB images across 102 classes, including `BACKGROUND_Google`.

The dataset is divided into three subsets:

| Split | Percentage | Images |
|---|---:|---:|
| Training | 70% | 6,400 |
| Validation | 10% | 914 |
| Testing | 20% | 1,830 |
| **Total** | **100%** | **9,144** |

### Preprocessing

- Resize images to 256 × 256.
- Centre crop to 224 × 224.
- Normalize using ImageNet statistics:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard deviation: `[0.229, 0.224, 0.225]`
- Apply random horizontal flipping and colour jitter during training.
- Batch size: 32.
- Random seed: 42.

## Model Architectures

### 1. ResNet-50

ResNet-50 uses residual connections:

`H(x) = F(x) + x`

- Feature dimension: 2048
- Approximate parameters: 23.7M
- Approximate computation: 4.1 GFLOPs

### 2. DenseNet-121

DenseNet-121 connects each layer to all preceding layers through feature concatenation:

`x_l = H([x_0, ..., x_(l-1)])`

- Feature dimension: 1024
- Approximate parameters: 7.06M
- Approximate computation: 2.9 GFLOPs

For both architectures, the original ImageNet classification head is replaced with a fully connected layer for 102 output classes. Pretrained backbone weights are retained.

## Fine-tuning Strategies

### Strategy 1: Standard Fine-tuning

The backbone is initially frozen, and selected higher-level layer groups are unfrozen.

Two configurations are evaluated for each architecture.

| Model | Configuration 1 | Configuration 2 |
|---|---|---|
| ResNet-50 | FC + Layer4 | FC + Layer4 + Layer3 |
| DenseNet-121 | Classifier + DenseBlock4 | Classifier + DenseBlock4 + Transition3 |

This approach preserves lower-level pretrained features while allowing higher-level representations to adapt to the target dataset.

### Strategy 2: Adaptive Fine-tuning (Filter Importance Ranking)

Instead of unfreezing complete layers, this approach identifies important convolutional filters using their average activation magnitude.

The importance score for a filter is:

`score(f) = mean(|output_f|)`

The scores are calculated across training batches and spatial positions.

Two variants are implemented:

**A. Last-Block Adaptive Fine-tuning**

- Calculate filter importance only within the final convolutional block.
- ResNet-50: Layer4.
- DenseNet-121: DenseBlock4.
- Select the top 25% of filters.
- Apply gradient masking to restrict updates to selected filters.

**B. Global Adaptive Fine-tuning**

- Calculate importance across all convolutional layers.
- Rank filters globally.
- Select the top 25%.
- Apply gradient masking to unselected filters.

Unselected filters receive zero gradient during backpropagation.

## Training Configuration

Hyperparameters were evaluated using a grid search.

| Parameter | Values |
|---|---|
| Optimizer | SGD, Adam |
| Learning rate | 1e-3, 1e-4, 1e-5 |
| Scheduler | StepLR, ReduceLROnPlateau |
| Grid-search duration | 5 epochs per configuration |
| Weight decay | 1e-4 |
| StepLR decay factor | 0.1 |
| StepLR step size | 10 epochs |
| Early stopping patience | 5 epochs |
| Batch size | 32 |

Selected configurations:

| Model | Optimizer | Learning Rate | Scheduler |
|---|---|---|---|
| ResNet-50 | Adam | 1e-4 | StepLR |
| DenseNet-121 | Adam | 1e-3 | StepLR |

The best checkpoint is selected using validation accuracy, with early stopping based on validation loss.

## Experimental Results

### 1. Standard Fine-tuning

| Metric | ResNet C1 | ResNet C2 | DenseNet C1 | DenseNet C2 |
|---|---:|---:|---:|---:|
| Trainable params | 15.2M | 22.3M | 2.26M | 2.79M |
| Test accuracy | 93.28% | 92.51% | 90.93% | **93.61%** |
| Macro F1 | 0.8991 | 0.8828 | 0.8678 | 0.8991 |
| Training time | 100.9s | 147.1s | 100.1s | 333.0s |

### 2. Adaptive Fine-tuning

| Metric | ResNet-50 | DenseNet-121 |
|---|---:|---:|
| Last-block test accuracy | 91.26% | 92.68% |
| Last-block trainable params | 15.2M | 1.87M |
| Last-block macro F1 | 0.8431 | 0.8952 |
| Global test accuracy | 85.52% | 84.59% |
| Global trainable params | 22.4M | 5.54M |

### Key Observations

- DenseNet-121 achieved the highest test accuracy of **93.61%** using standard fine-tuning.
- ResNet-50 achieved **93.28%** test accuracy with a training time of 100.9 seconds.
- DenseNet-121 achieved 92.68% accuracy with only 1.87M trainable parameters using last-block adaptive fine-tuning.
- Global adaptive fine-tuning performed poorly for both architectures.
- Unfreezing too many parameters increased overfitting, particularly for ResNet-50.

## Conclusion

Standard fine-tuning outperformed adaptive filter-importance-based fine-tuning across the evaluated configurations.

DenseNet-121 demonstrated better parameter efficiency and achieved the highest overall test accuracy. Its dense connectivity allows feature reuse across layers, making last-block adaptation particularly effective.

ResNet-50 provided a competitive speed-accuracy trade-off with fewer training time requirements in its best configuration.

Overall, the experiments highlight the importance of considering architectural differences when designing parameter-efficient fine-tuning strategies.

## Limitations and Future Work

- Caltech-101 has considerable class imbalance.
- Activation magnitude may not always represent task-specific filter importance.
- Global filter ranking can select filters that are not necessarily useful for the target task.
- Macro F1 should be interpreted cautiously due to class imbalance.

Potential future improvements include:

- Class-balanced sampling and loss reweighting.
- Alternative filter-importance measures.
- Combining activation-based ranking with gradient-based importance.
- Evaluating on additional image classification datasets.

## Reproducibility

The experiments use a fixed random seed (42), a consistent dataset split, and documented training configurations.

To reproduce the experiments, the repository should include:

- Dataset preparation instructions.
- Model initialization and classification-head modification.
- Standard fine-tuning configurations.
- Adaptive filter-importance calculation and gradient masking.
- Training and evaluation scripts.
- Saved configuration files and checkpoints, where available.

**Environment and execution instructions:** To be completed with the actual implementation's dependency versions, repository structure, and training entry points.

## Hardware

Experiments were conducted on:

- GPU: NVIDIA RTX 4000 Ada Generation
- Batch size: 32

## Acknowledgements

- Caltech-101 dataset.
- ImageNet pretrained model architectures: ResNet-50 and DenseNet-121.
