import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
l_train = pd.read_csv("/Users/zhangyunping/PycharmProjects/yolov5holo/experiment_record_paper/"
                      "run-exp_512_50_particle2-tag-train_cls_loss.csv")
l_val = pd.read_csv("/Users/zhangyunping/PycharmProjects/yolov5holo/experiment_record_paper/"
                    "run-exp_512_50_particle2-tag-val_depth_loss.csv")


l_dep = pd.read_csv("/Users/zhangyunping/PycharmProjects/yolov5holo/experiment_record_paper/"
                      "run-exp_512_50_particle2-tag-train_cls_loss.csv")
l_obj = pd.read_csv("/Users/zhangyunping/PycharmProjects/yolov5holo/experiment_record_paper/"
                      "run-exp_512_50_particle2-tag-train_obj_loss.csv")
l_box = pd.read_csv("/Users/zhangyunping/PycharmProjects/yolov5holo/experiment_record_paper/"
                      "run-exp_512_50_particle2-tag-train_box_loss.csv")

step = l_train['Step'][1:51]
l_t_d = l_dep['Value'][1:51]
l_t_o = l_obj['Value'][1:51]
l_t_b = l_box['Value'][1:51]

l_v = l_val['Value'][1:51]*0.1

fig = plt.figure()
ax = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax.plot(step, l_t_d,'g^-',label = 'Depth regression loss',markersize = 3)
ax2.plot(step, l_t_o,'g^-',label = 'Objectiveness loss',markersize = 3)
ax3.plot(step, l_t_b,'g^-',label = 'Bounding box regression loss',markersize = 3)
# ax.plot(step,l_v,'r.-',label = 'validation',markersize = 4)

xticks =np.array(range(0,60,10))
ax.set_xticks(xticks)
ax2.set_xticks(xticks)
ax3.set_xticks(xticks)
for x in xticks:
    ax.axvline(x=x, lw=0.25, ls="--",c='k')  # 添加垂直直线
    ax2.axvline(x=x, lw=0.25, ls="--", c='k')  # 添加垂直直线
    ax3.axvline(x=x, lw=0.25, ls="--", c='k')  # 添加垂直直线


ax.set_xlabel('Epochs')
ax.set_xlim(0,50)
ax.set_ylabel('Loss')
ax.legend()

ax2.set_xlabel('Epochs')
ax2.set_xlim(0,50)
ax2.set_ylabel('Loss')
ax2.legend()

ax3.set_xlabel('Epochs')
ax3.set_xlim(0,50)
ax3.set_ylabel('Loss')
ax3.legend()
