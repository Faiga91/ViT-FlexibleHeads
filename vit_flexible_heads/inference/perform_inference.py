"""
This script demonstrates how to use LazyConfig 
to load a configuration file and run inference 
on a dataset.
"""

import os
import logging

from tqdm import tqdm
import torch
from torch.utils.data import SequentialSampler, DataLoader, default_collate
import cv2

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import transforms as T
from detectron2.data import DatasetFromList

from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def preprocess_image(cfg, image_path):
    """
    Preprocess an image for inference.

    Args:
        cfg: Configuration object containing preprocessing details.
        image_path: Path to the input image.

    Returns:
        inputs: Preprocessed image tensor and metadata for inference.
        image: Original image in RGB format.
    """
    image = cv2.imread(image_path)  # pylint: disable=no-member
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = image[:, :, ::-1]  # Convert BGR to RGB

    height, width = image.shape[:2]
    augmentations = [
        instantiate(aug) for aug in cfg.dataloader.test.mapper.augmentations
    ]
    augmentation = T.AugmentationList(augmentations)

    aug_input = T.AugInput(image)
    augmentation(aug_input)

    image_transformed = aug_input.image
    image_tensor = torch.as_tensor(
        image_transformed.transpose(2, 0, 1).astype("float32")
    )

    pixel_mean = torch.Tensor(cfg.model.pixel_mean).reshape(-1, 1, 1)
    pixel_std = torch.Tensor(cfg.model.pixel_std).reshape(-1, 1, 1)
    image_normalized = (image_tensor - pixel_mean) / pixel_std

    inputs = {"image": image_normalized, "height": height, "width": width}

    return inputs, image


def run_inference(model, inputs, device):
    """
    Run inference using the provided model.

    Args:
        model: The model instance for inference.
        inputs: Preprocessed input for the model.
        device: Device to run inference on.

    Returns:
        outputs: Model predictions.
    """
    with torch.no_grad():
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        outputs = model([inputs])[0]
    return outputs


def visualize_predictions(cfg, image, outputs):
    """
    Visualize model predictions on an image.

    Args:
        cfg: Configuration object containing metadata details.
        image: Original image in RGB format.
        outputs: Model predictions.

    Returns:
        result_image: Image with visualized predictions.
    """
    dataset_name = cfg.dataloader.test.dataset.names
    metadata = MetadataCatalog.get(dataset_name)

    visualizer = Visualizer(image, metadata=metadata, scale=1.0)
    instances = outputs["instances"].to("cpu")
    vis_output = visualizer.draw_instance_predictions(instances)
    result_image = vis_output.get_image()
    return result_image


if __name__ == "__main__":
    # Paths
    ## Hyper-Kvasir
    ANNO_PATH = "/dataset/hyper-kvasir/test-COCO-annotations.json"
    IMG_DIR = "/dataset/hyper-kvasir/test"

    CONFIG_FILE = "../experiments/configs/mask_rcnn_vitdet_config.py"
    WEIGHTS_PATH = "output/model_final.pth"
    OUT_DIR = "inference"

    # Dataset registration
    register_coco_instances("my_custom_test_dataset", {}, ANNO_PATH, IMG_DIR)
    meta_data = MetadataCatalog.get("my_custom_test_dataset")
    logger.info("Dataset Metadata:%s", meta_data)

    # Load configuration and model
    curr_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LazyConfig.load(CONFIG_FILE)

    config.model.roi_heads.num_classes = 1
    config.model.roi_heads.box_predictor.test_score_thresh = 0.5
    config.train.init_checkpoint = WEIGHTS_PATH
    config.train.device = str(curr_device)
    config.dataloader.test.dataset.names = "my_custom_test_dataset"
    config.dataloader.num_workers = 4
    config.dataloader.test.total_batch_size = 2

    vitDet_model = instantiate(config.model).to(curr_device)
    DetectionCheckpointer(vitDet_model).load(config.train.init_checkpoint)

    vitDet_model.eval()

    augs = [instantiate(aug) for aug in config.dataloader.test.mapper.augmentations]
    mapper = DatasetMapper(is_train=False, augmentations=augs, image_format="BGR")
    dataset_dicts = DatasetCatalog.get("my_custom_test_dataset")

    sampler = SequentialSampler(dataset_dicts)

    mapped_dataset = DatasetFromList(dataset_dicts, copy=False)
    mapped_dataset = list(map(mapper, mapped_dataset))

    data_loader = DataLoader(
        dataset=mapped_dataset,
        batch_size=config.dataloader.test.total_batch_size,
        num_workers=config.dataloader.num_workers,
        collate_fn=default_collate,  # Detectron2's collator for batching
    )
    logger.info("Data Loader: %s", data_loader)

    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Get list of image files
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = [
        os.path.join(IMG_DIR, f)
        for f in os.listdir(IMG_DIR)
        if f.lower().endswith(image_extensions)
    ]

    # Run inference using dataloader
    for batch in tqdm(data_loader, desc="Processing images"):
        images = batch["image"]
        logger.info("Processing batch of %d images", len(images))
        logger.debug("Images: %s", images)
        file_names = batch.get("file_name", ["unknown"] * len(images))
        logger.debug("File names: %s", file_names)
        ins = [
            {"image": image, "height": image.shape[1], "width": image.shape[2]}
            for image in images
        ]
        outs = vitDet_model(ins)
        logger.info("Outputs: %s", outs)

        for i, output in enumerate(outs):
            logger.info("Processing file: %s", file_names[i])

            # Extract the original image for visualization
            original_image = images[i].permute(1, 2, 0).cpu().numpy()

            if "instances" in output and len(output["instances"]) > 0:
                # Visualize predictions
                res_image = visualize_predictions(config, original_image, output)

                # Save the visualized result
                output_path = os.path.join(OUT_DIR, os.path.basename(file_names[i]))
                cv2.imwrite(output_path, res_image)  # pylint: disable=no-member
                logger.info("Saved visualized output to %s", output_path)
            else:
                logger.info("No instances detected in %s", file_names[i])
