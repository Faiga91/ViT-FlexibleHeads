# ViT-FlexibleHeads
Vision Transformer with Flexible Heads. 


## Install Dependencies

Detectron2 is the primary dependency for this project. Follow the steps below to install it:

1. You can install Detectron2 either from source or via pip.
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
