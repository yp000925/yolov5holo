#!/bin/bash

python train.py \
--weights '' \
--cfg 'yolov5s.yaml' \
--data 'holodata.yaml' \
--hyp 'data/hyp_self.yaml' \
--epochs 15 \
--batch-size 16 \
--img-size [512, 512] \
--adam \
--name 'exp00' \
--save_period 50 \
