import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm
from test import pred_label_onehot

from models.experimental import attempt_load
from utils.datasets import create_dataloader,create_dataloader_modified
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr,post_nms, nms_modified
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt,plot_images_modified
from utils.torch_utils import select_device, time_synchronized

import torchvision



if __name__ == '__main__':

    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    source = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/small_test'
    # weights = '/Users/zhangyunping/PycharmProjects/yolov5holo/train/exp3/best.pt'
    weights = '/Users/zhangyunping/PycharmProjects/yolov5holo/train/exp_depthmap/best.pt'
    view_image = True
    img_size = 512
    # project = '/content/drive/MyDrive/yoloV5/train/exp3'
    task = 'test'
    device = torch.device('cpu')
    set_logging()
    batch_size = 32

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(img_size, s=gs)  # check img_size

    # half = device.type != 'cpu'  # half precision only supported on CUDA
    # if half:
    #     model.half()

    model.eval()
    nc = 256 # number of classes

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader_modified(source, imgsz, batch_size, gs,
                                    pad=0.5, rect=True,
                                   prefix=colorstr(f'{task}: '),image_weights=True)[0]

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}


    mat = ConfusionMatrix(nc=256, conf=0.5, iou_thres=0.6)


    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        # targets in the format [batch_idx, class_id, x,y,w,h]
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        targets = targets.to(device)
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels

        out, train_out = model(img)  # inference and training outputs

        # # if would like to use one_hot for output
        # out = pred_label_onehot(out)
        # out = non_max_suppression(out)
        # out = post_nms(out,0.45)# list of anchors with [xyxy, conf, cls]


        # # if would like to use depthmap as the class directly
        out = nms_modified(out,obj_thre=0.8, iou_thres=0.5, nc=256) # list of anchors with [xyxy, conf, cls]
        # 因为用了torch自带的nms所以变成了xyxy


# plot ----------------------------------------------------------------------------------------------------------------
        # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        plot_images_modified(img, targets, paths, fname='check.jpg', names=None)
        plot_images_modified(img, output_to_target(out),paths ,fname = 'check_pred.jpg',names=None)


# update confusion matrix ----------------------------------------------------------------------------------------------
        for batch_idx in range(len(out)):
            labels = targets[targets[:,0].int()==batch_idx][:,1::] # class, x,y,w,h
            detections = out[batch_idx] # x,y,x,y,conf,cls
            detections[:,5] = detections[:,5].int()
            labels[:,1::] = xywh2xyxy(labels[:,1::])# class, x,y,x,y
            mat.process_batch(detections,labels)

