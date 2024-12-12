#!/bin/bash
python finetune_detection_head.py \
--config configs/mask_rcnn_vitdet_config.py \
--eval-only \
train.init_checkpoint=output/model_final.pth \
> logs/ViTDet_eval.txt 2>&1