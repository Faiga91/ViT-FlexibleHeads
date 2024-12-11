#!/bin/bash
cd ../evaluation
python test_detection_head.py \
> ../experiments/logs/test_detection_head.txt 2>&1
