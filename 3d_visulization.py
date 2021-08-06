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
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr,post_nms, nms_depthmap
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt,plot_images_modified
from utils.torch_utils import select_device, time_synchronized
import matplotlib.pyplot as plt
import torchvision
import matplotlib
matplotlib.use('TkAgg')

def plot_3d(labels,img_size):
    z = labels[:,0]
    x = (labels[:,1]+0.5*labels[:,3])*img_size
    y = (labels[:,2]+0.5*labels[:,4])*img_size
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(np.array(x),np.array(y),np.array(z),c='r',marker='o')
    plt.show()


def get_xyz(bbox,z,xy_size,z_res=1):
    # bbox x,y,w,h
    center_x = round((bbox[0]+0.5*bbox[2])*xy_size)
    center_y = round((bbox[1]+0.5*bbox[3])*xy_size)
    z = round(z/z_res)
    return (center_x,center_y,z)

if __name__ == '__main__':

    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    source = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/small_test'
    weights = '/Users/zhangyunping/PycharmProjects/yolov5holo/train/exp_bboxrescaled_weightedloss/best.pt'
    view_image = True
    img_size = 512
    # project = '/content/drive/MyDrive/yoloV5/train/exp3'
    task = 'test'
    device = torch.device('cpu')
    set_logging()
    batch_size = 8

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


    mat = ConfusionMatrix(nc=256, conf=0, iou_thres=0.6) #conf should change accordingly 0.8 for depthmap


    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        # targets in the format [batch_idx, class_id, x,y,w,h]
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width

        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
        labels = targets[:, 1::]
        out, train_out = model(img)
        out = nms_depthmap(out, obj_thre=0.8, iou_thres=0.5, nc=256)
        for batch_idx in range(len(out)):
            labels = targets[targets[:,0].int()==batch_idx][:,1::] # class, x,y,w,h
            detections = out[batch_idx] # x,y,x,y,conf,cls
            # detections[:,5] = detections[:,5].int()
            preds = torch.zeros((detections.shape[0],5))
            preds[:,1::] = xyxy2xywh(detections[:,0:4])
            preds[:,0] = detections[:,5]
            # labels[:,1::] = xywh2xyxy(labels[:,1::])# class, x,y,x,y
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

            for c, m, plot_data in [('r', 'o',labels), ('b', '^', preds)]:
                z = plot_data[:, 0]
                x = (plot_data[:, 1] + 0.5 * plot_data[:, 3])
                y = (plot_data[:, 2] + 0.5 * plot_data[:, 4])
                ax.scatter(np.array(x), np.array(y), np.array(z), c=c, marker=m)
            plt.show()
        break
