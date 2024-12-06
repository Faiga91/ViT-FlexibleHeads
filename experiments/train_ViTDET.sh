#!/bin/bash
python finetune_detection_head.py \
--config mask_rcnn_vitdet_config.py \
> logs/fintune_detection_head.txt 2>&1
