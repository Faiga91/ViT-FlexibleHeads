# ViT-FlexibleHeads
[![Python 3.8](https://img.shields.io/badge/python-=%3E3.8-blue.svg)](https://www.python.org/downloads/release/python-3816/)
[![Pylint](https://github.com/Faiga91/ViT-FlexibleHeads/actions/workflows/pylint.yml/badge.svg)](https://github.com/Faiga91/ViT-FlexibleHeads/actions/workflows/pylint.yml)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://faiga91.github.io/ViT-FlexibleHeads)


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
```

## How to use? 
Navigate to the experiments folder, which contains the scripts for finetunning, inference and testing.

1. The `train_ViTDET.sh` script serves as the main entry point for fine-tuning the object detection head of a Vision Transformer (ViTDet) model using a Mask R-CNN configuration. This script encapsulates all the steps required to start the training process with minimal setup.


    1a. Make the script execuatble 
    ```bash
    chmod +x train_ViTDET.sh
    ```
    1.b Run the script 

    ```bash
    ./train_ViTDET.sh
    ```

2. The configuration file `vit_flexible_heads/experiments/configs/mask_rcnn_vitdet_config.py`  sets up a training environment for a custom object detection model using a Vision Transformer (ViT) backbone integrated into Detectron2’s Mask R-CNN framework.
    2.1 Dataset Registeration: 
    ```python
    register_coco_instances("my_custom_dataset_train", {}, "/path/to/train-annotations.json", "/path/to/train-images")
    register_coco_instances("my_custom_dataset_val", {}, "/path/to/val-annotations.json", "/path/to/val-images")
    ```
    Make sure these paths point to your dataset’s COCO-style annotation files and image directories.

    2.2 Number of Classes:
    ```python
    model.roi_heads.num_classes = 1
    ```
    Set this to the number of object classes you want to detect.

    2.3 Predictor Settings:
    ```python
    test_score_thresh=0.05
    test_nms_thresh=0.5
    ```
    Control the score threshold and Non-Maximum Suppression (NMS) threshold during inference.

    2.4 Initialization & Iterations:
    ```python
    train["init_checkpoint"] = "path/to/pretrained/vit_weights.pth"
    train["max_iter"] = 1000
    ```
    We set init_checkpoint to pretrained weights of the ViT backbone, and choose a suitable number of training iterations.

    2.5 Learning Rate & Schedule:
    ```python
    lr_multiplier = WarmupParamScheduler(
    scheduler=MultiStepParamScheduler(values=[1.0, 0.1, 0.01], milestones=[500, 800]),
    warmup_length=2 / train["max_iter"],
    warmup_factor=0.001,
    )
    ```
    This uses a warmup phase and multi-step downscaling of LR over training. Tweak milestones and values based on your dataset and desired training regime.

    2.6 Optimizer
    ```python
    optimizer = torch.optim.AdamW(
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    )
    ```
    Adjust lr, betas, and weight_decay as needed.
    








