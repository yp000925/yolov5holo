import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader,create_dataloader_modified
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt,plot_images_modified
from utils.torch_utils import select_device, time_synchronized

import torchvision


def post_nms(pred, iou_thre):
    output = [torch.zeros((0, 6), device=pred.device)] * pred.shape[0]
    nc = pred.shape[2] - 5
    xc = pred[..., 4] > 0.8  # objectiveness

    for xi, x in enumerate(pred):
        x = x[xc[xi]]  # #cls_conf>thred, nc+5 [bbox,objectiveness,nc] nc代表的每个class的confidence
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)
        # x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thre)
        output[xi] = x[i]
    return output

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
    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        out, train_out = model(img)  # inference and training outputs
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        # out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        out = post_nms(out,0.45)
        # list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        plot_images_modified(img, targets, paths, fname='check.jpg', names=None)
        plot_images_modified(img,output_to_target(out),paths ,fname = 'check_pred.jpg',names=None)