# ViT-FlexibleHeads
[![Python 3.8](https://img.shields.io/badge/python-=%3E3.8-blue.svg)](https://www.python.org/downloads/release/python-3816/)
[![Pylint](https://github.com/Faiga91/ViT-FlexibleHeads/actions/workflows/pylint.yml/badge.svg)](https://github.com/Faiga91/ViT-FlexibleHeads/actions/workflows/pylint.yml)

Vision Transformer with Flexible Heads. 


## Install Dependencies
- Install PyTorch from the official website [here](https://pytorch.org/get-started/locally/)
- Detectron2 is the primary dependency for this project. Follow the steps below to install it:

1. You can install Detectron2 from the source, i.e., the Github repo.
2. To address most of the future warnings, you can use our custom fork by running the following command:

```
python -m pip install 'git+https://github.com/Faiga91/detectron2.git'
```

3. For more detailed installation instructions, refer to the official Detectron2 documentation available [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

### Additional Dependencies

Ensure all additional dependencies listed in the `requirements.txt` file are installed to avoid compatibility issues. You can install them by running:

```
pip install -r requirements.txt
```

## Project Structure 
The repository is organized as follows: 

```plaintext
ViT-FlexibleHeads/
├── datasets_loaders/        # Dataset loading and preprocessing utilities
├── evaluation/              # Scripts for model evaluation
├── experiments/             # Experiment setups and configurations
├── inference/               # Inference scripts for running predictions
├── models/                  # Model definitions and utilities
├── .gitignore               # Git ignore rules
├── .pre-commit-config.yaml  # Pre-commit hook configuration
├── LICENSE                  # Project license
├── README.md                # Project overview
├── requirements.txt         # Python dependencies

## How to use? 
Navigate to the experiments folder, which contains the scripts for finetunning, inference and testing.

1. The `train_ViTDET.sh` script serves as the main entry point for fine-tuning the object detection head of a Vision Transformer (ViTDet) model using a Mask R-CNN configuration. This script encapsulates all the steps required to start the training process with minimal setup.

1a. Make the script execuatble 

```
chmod +x train_ViTDET.sh
```
1.b Run the script 

```
./train_ViTDET.sh
```
