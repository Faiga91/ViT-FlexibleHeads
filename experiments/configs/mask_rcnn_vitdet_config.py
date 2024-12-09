"""
Based on DETECTRON2's mask_rcnn_vitdet_b_100p.py
"""

from functools import partial
import torch
from torch import nn
import sys
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.data.datasets import register_coco_instances
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.modeling import ViT, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.solver.build import get_default_optimizer_params


sys.path.append("..")
from datasets_loaders.coco_loader_lsj import dataloader
from models.mask_rcnn_fpn import model
from models.constants import constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 1
# pylint: disable=line-too-long
BACKBONE_PATH = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"

register_coco_instances(
    "my_custom_dataset_train",
    {},
    "/dataset/hyper-kvasir/train-COCO-annotations.json",
    "/dataset/hyper-kvasir/train",
)
register_coco_instances(
    "my_custom_dataset_val",
    {},
    "/dataset/hyper-kvasir/val-COCO-annotations.json",
    "/dataset/hyper-kvasir/val",
)


dataloader.train.dataset.names = "my_custom_dataset_train"
dataloader.test.dataset.names = "my_custom_dataset_val"

model.pixel_mean = constants["imagenet_rgb256_mean"]
model.pixel_std = constants["imagenet_rgb256_std"]
model.input_format = "RGB"

# Base
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(ViT)(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=1024,
)

model.roi_heads.box_head.conv_norm = "LN"
model.roi_heads.mask_in_features = []
model.roi_heads.mask_head = None

model.roi_heads.num_classes = 1
model.roi_heads.box_head.fc_dims = [512]

box2box_transform = Box2BoxTransform(weights=(10, 10, 5, 5))
model.roi_heads.box_predictor = FastRCNNOutputLayers(
    input_shape=512,
    num_classes=model.roi_heads.num_classes,
    box2box_transform=box2box_transform,
    test_score_thresh=0.05,
    test_nms_thresh=0.5,
)

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [512]

model.roi_heads.box_predictor = FastRCNNOutputLayers(
    input_shape=512,
    num_classes=model.roi_heads.num_classes,
    box2box_transform=box2box_transform,
    test_score_thresh=0.05,
    test_nms_thresh=0.5,
)

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [512]

model.roi_heads.num_classes = NUM_CLASSES
model.roi_heads.mask_head = None
model.roi_heads.mask_in_features = None
model.roi_heads.mask_pooler = None

# Initialization and trainer settings
train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter=90000,
    amp=dict(enabled=False),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=5000, max_to_keep=100),  # options for PeriodicCheckpointer
    eval_period=5000,
    log_period=20,
    device="cuda",
)

train["amp"]["enabled"] = True
train["ddp"]["fp16_compression"] = True
train["init_checkpoint"] = BACKBONE_PATH

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
# train.max_iter = 184375
train["max_iter"] = 10

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        # milestones=[163889, 177546],
        milestones=[5, 8],
        num_updates=train["max_iter"],
    ),
    warmup_length=2 / train["max_iter"],
    warmup_factor=0.001,
)

# Optimizer
optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        base_lr="${..lr}",
        weight_decay_norm=0.0,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)

optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
