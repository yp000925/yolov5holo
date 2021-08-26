
import sys
sys.path.insert(0,'/Users/zhangyunping/PycharmProjects/yolov5holo')

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
import pandas as pd
#
# m = 1.0
# cm=1e-2*m
# mm=1e-3*m
# um=1e-6*m
# nm=1e-9*m
#
# wavelength = 633 * nm
# N = 512
# # pixel_pitch = 10*um
# frame = 10 * mm  # 10mm * 10mm
# # size_range = [20 * um, 100 * um]
# size_range = [50*um,50*um]
# res_z =  (3*cm-1*cm)/256
#
# def get_label(a):
#     px = int(a[0]/frame*N+N/2)
#     py = int(N/2+a[1]/frame*N)
#     pz = a[2]
#     # buffer = p_size*10
#     buffer = 30
#     bbox_x = max(0, px-buffer)
#     bbox_y = max(0, py-buffer)
#     height = buffer*2
#     width = buffer*2
#     if bbox_x+width > N:
#         width = N-bbox_x
#     if bbox_y+height > N:
#         height = N-bbox_y
#     seg = [bbox_x,bbox_y,bbox_x,bbox_y+height,bbox_x+width,bbox_y+height,bbox_x+width,bbox_y]
#     return (pz,bbox_x,bbox_y,width,height)
#
# def plot_3d_csv(labels):
#     z = labels[:,0]*100
#     x = (labels[:,1]+0.5*labels[:,3])
#     y = (labels[:,2]+0.5*labels[:,4])
#     fig = plt.figure()
#     ax = fig.add_subplot(111,projection = '3d')
#     ax.set_xlim(0, 512)
#     ax.set_ylim(0, 512)
#     ax.set_xlabel('X ')
#     ax.set_ylabel('Y ')
#     ax.set_zlabel('Depth(cm)')
#     ax.scatter(np.array(x),np.array(y),np.array(z),c='r',marker='o')
#     plt.show()
#
#
# holo_csv = '/Users/zhangyunping/PycharmProjects/yolov5holo/experiment_record_paper/0.csv'
# holo_label = pd.read_csv(holo_csv)
# arr = np.array(holo_label)
# # x = holo_label['x']
# # y = holo_label['y']
# # z = holo_label['z']
# # size = holo_label['size']
# labels = [get_label(x) for x in arr]
# labels = np.array(labels)
#
# plot_3d_csv(labels)

def plot_3d_ouput(labels,ax,color,marker):
    z = labels[:,0]/255.0 #class,x1,y1,x2,y2
    x = (labels[:,1]+labels[:,3])/2
    y = (labels[:,2]+labels[:,4])/2
    ax.scatter(np.array(x),np.array(y),np.array(z),c=color,marker=marker)

correct = np.load("/Users/zhangyunping/PycharmProjects/yolov5holo/experiment_record_paper/correct_20.npy")
# miss = np.load("/Users/zhangyunping/PycharmProjects/yolov5holo/experiment_record_paper/miss_150.npy")

fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.set_xlim(0, 512)
ax.set_ylim(0, 512)
ax.set_zlim(0, 1)
ax.set_xlabel('X(pixels) ')
ax.set_ylabel('Y(pixels) ')
ax.set_zlabel('Normalized depth')


# plot_3d_ouput(miss,ax,'r','o')
plot_3d_ouput(correct,ax,'g','o')

#%% draw the figure of precision and recall at differet densities
recall = [0.9435,0.9134,0.9180, 0.8873,0.8003]
precision = [1,1,1,1,1]
x = np.array([4,10,20,40,60])/10
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0, 1.1)
ax.plot(x,recall,'-o',c='green',label ='recall')
ax.plot(x,precision,'-o',c='blue',label ='precision')


errors = np.array([2.75,3.18,3.70,7.55,12.28])/256*2*10
ax2 = ax.twinx()
ax2.set_ylim(0, 1.1)
ax2.plot(x,errors,'-o',c='black',label ='depth error')


#%% draw the figure of precision and recall at different noise level
import matplotlib.pyplot as plt
import torchvision
import matplotlib
matplotlib.use('TkAgg')

# recall = [0.9172,0.9214,0.9201,0.9155,0.9175,0.9163,0.9143,0.9185,0.9275,0.9243]
#
# precision = [1,1,1,1,1,0.9989,1,0.9822,0.9468,0.8852]
# x = np.array([5,10,15,20,25,30,35,40,50,60])

recall = [0.9172,0.9214,0.9201,0.9155,0.9175,0.9163]

precision = [1,1,1,1,1,0.9989]
x = np.array([5,10,15,20,25,30])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim(0, 1.1)
ax.plot(x,recall,'-o',c='green',label ='recall')
ax.plot(x,precision,'-o',c='blue',label ='precision')


# errors = np.array([5.37,5.58,6.398,7.64,8.86,10.99,12.64,13.69,16.178,19.02])/256*2*10

errors = np.array([5.37,5.58,6.398,7.64,8.86,10.99])/256*2*10
ax2 = ax.twinx()
ax2.set_ylim(0.2, 2.1)
ax2.plot(x,errors,'-o',c='black',label ='depth error')
