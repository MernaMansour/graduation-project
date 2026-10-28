# Facial Emotion Recognition Using Deep Learning

A collection of Jupyter notebooks exploring **convolutional neural networks (CNNs)** and **transfer learning** for seven-class facial emotion recognition. The repository contains experiments with different image-classification architectures and evaluation workflows.

## Models and notebooks

| Notebook | Architecture |
|---|---|
| `CNN_high_accuracyipynb.ipynb` | CNN experiment |
| `Densenet.ipynb` | DenseNet |
| `Resnnet.ipynb` | ResNet |
| `VGG16_.ipynb` | VGG16 |
| `VGG16_1.ipynb` | VGG16 experiment |
| `efficientnet.ipynb` | EfficientNetB0 |
| `efficientnetB7S.ipynb` | EfficientNetB7 experiment |
| `mobilet.ipynb` | MobileNet experiment |

> The model names above are based on repository filenames. The detailed implementation and measured results were verified only for the supplied `efficientnet.ipynb` notebook.

## Classification task

The EfficientNet notebook uses seven emotion labels:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Verified EfficientNetB0 experiment

The supplied `efficientnet.ipynb` uses an ImageNet-pretrained **EfficientNetB0** backbone with the original classification head removed and the backbone layers frozen. It adds global average pooling, a 512-unit ReLU dense layer, dropout (0.5), and a seven-class softmax output.

| Configuration | Value |
|---|---|
| Image size | 224 × 224 RGB |
| Dataset split | 80% training / 10% validation / 10% testing |
| Training samples | 14,000 |
| Validation samples | 1,750 |
| Test samples | 1,750 |
| Batch size | 128 |
| Training epochs | 50 |
| Optimizer | Adam |
| Loss function | Categorical cross-entropy |
| Reported test accuracy | **88.91%** |
| Reported test loss | **0.3892** |

The notebook also includes plots of training/validation loss and accuracy, a normalized confusion matrix, and a classification report. These results are the saved notebook outputs, **not an independently reproduced benchmark**.

## Technologies

- Python and Jupyter Notebook / Google Colab
- TensorFlow / Keras
- NumPy and pandas
- Matplotlib and Seaborn
- scikit-learn
- `split-folders`

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/MernaMansour/YOUR_REPOSITORY_NAME.git
   cd YOUR_REPOSITORY_NAME
   ```

2. Open a notebook in Google Colab or your preferred Jupyter environment.
3. Prepare the facial-emotion image dataset in class-specific folders.
4. Update the dataset and Google Drive paths to match your environment.
5. Install the packages required by the notebook, then run cells in order.

**Note:** The provided EfficientNet notebook expects a ZIP archive in a particular Google Drive location. The dataset is not included in the repository listing shown, and some older Keras APIs in the notebook (such as `fit_generator` and `predict_generator`) may need updating for current TensorFlow/Keras versions.

## Repository structure

```text
├── CNN_high_accuracyipynb.ipynb
├── Densenet.ipynb
├── README.md
├── Resnnet.ipynb
├── VGG16_.ipynb
├── VGG16_1.ipynb
├── efficientnet.ipynb
├── efficientnetB7S.ipynb
└── mobilet.ipynb
```

## Project goals

This repository is intended to help explore CNN architectures, transfer learning, model training, and evaluation for facial-expression classification. Model comparisons should use the same dataset splits and evaluation protocol before drawing conclusions about relative performance.

## Author

**Merna Mansour**  
GitHub: [@MernaMansour](https://github.com/MernaMansour)
