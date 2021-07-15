from utils.datasets import create_dataloader_modified
from utils.autoanchor import *



data_source = '/Users/zhangyunping/PycharmProjects/Holo_synthetic/datayoloV5format/images/small_test'
img_size = 512

batch_size = 128
anchors = kmean_anchors(path='test_data.yaml', n=9, img_size=512, thr=4.0, gen=1000, verbose=True)


#
# dataloader, dataset = create_dataloader_modified(data_source, img_size, batch_size)
#
# # project = '/content/drive/MyDrive/yoloV5/train/exp3'
#
# for batch_i, (img, targets, paths,_) in enumerate(dataloader):
#     img = img.float()
#     img /= 255.0
#
#     wh = targets[:, 4:] * img_size
#
#     break

