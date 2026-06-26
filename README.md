# Brain Tumor Detection using MRI Data

This repository contains an experimental PyTorch-based pipeline for binary classification of brain MRI scans. The primary artifact is a Jupyter notebook that loads raw image data, constructs a custom dataset, defines convolutional network variants, and evaluates performance with basic validation metrics.

## Project Overview

The project explores the problem of distinguishing healthy brain MR images from tumor-positive MRI scans. The dataset is organized into two classes (`yes`, `no`) and the notebook demonstrates a hands-on approach to image preprocessing, model definition, training, and visual analysis.

This repository is intended for a technical audience and demonstrates a sequence of empirical experiments rather than a polished production system.

## Repository Layout

- `brain_tumors_detection.ipynb` — primary notebook containing data ingestion, model definition, training loops, and evaluation.
- `requirements_new.txt` — dependency manifest covering scientific computing, image processing, deep learning, and Jupyter tooling.
- `data/brain_tumor_dataset/yes` — MRI images labeled as tumor present.
- `data/brain_tumor_dataset/no` — MRI images labeled as tumor absent.

## Dataset Source

The notebook references the publicly available dataset from Kaggle:

`https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection/code`

The repository assumes the dataset is already extracted into the `data/brain_tumor_dataset` folder with separate `yes` and `no` subdirectories.

## Execution Flow

The notebook proceeds through the following stages:

1. Image ingestion and normalization
2. Construction of a custom `torch.utils.data.Dataset`
3. Visualization of sample MRI images for manual inspection
4. Definition of multiple convolutional neural network variants
5. Model evaluation before and after training
6. Training with binary cross-entropy loss
7. Confusion matrix generation and accuracy computation
8. Train/validation split and overfitting analysis

## Model Architecture

The notebook defines several model variants, including:

- `CNN` — a minimal convolutional classifier with two convolutional blocks and fully connected layers.
- `CNN_V2` — a deeper architecture using batch normalization, ReLU activations, and dropout.
- `CNN_V3` — an expanded architecture with additional convolutional stages and a larger fully connected head.

The models use sigmoid activation for binary classification and are trained with `torch.optim.Adam`.

## Environment and Dependencies

The required dependencies are listed in `requirements_new.txt`. Key packages include:

- `numpy`, `pandas`, `scipy` — core scientific libraries
- `torch`, `torchvision` — deep learning framework
- `opencv-python` — image loading and preprocessing
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — dataset splitting and evaluation metrics
- `jupyter`, `ipykernel` — notebook execution

## Setup Instructions

1. Create a clean Python environment.
2. Install dependencies from `requirements_new.txt`.
3. Ensure the dataset is extracted under `data/brain_tumor_dataset/yes` and `data/brain_tumor_dataset/no`.
4. Open `brain_tumors_detection.ipynb` in JupyterLab or Jupyter Notebook.
5. Execute the notebook sequentially.

Example:

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements_new.txt
jupyter lab
```

## Notes on Reproducibility

- The notebook uses fixed image resizing to `128x128` pixels.
- Class labels are encoded as `0` for healthy and `1` for tumor.
- Validation is performed with an 80/20 split, and the code uses `random_state=42` for reproducibility.

## Practical Considerations

This implementation is intentionally exploratory. It is suitable for demonstration, debugging, and rapid prototyping, but it does not include:

- explicit data augmentation,
- advanced learning rate scheduling,
- a robust training pipeline,
- checkpoint persistence,
- production-grade preprocessing.

For a more rigorous study, the notebook should be extended with proper data loaders, augmentation, early stopping, and a dedicated training script.

## Contact

This repository is structured for experimentation and algorithmic study. If additional refinement is required, the next logical step is to extract the training logic into modular Python scripts and to create a consistent evaluation pipeline with separate train/validation/test splits.
