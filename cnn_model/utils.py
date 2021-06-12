import os
from os.path import basename
from pathlib import Path
from shutil import copyfile

import matplotlib
# matplotlib.use('Agg')
import matplotlib.cm
import numpy as np
import matplotlib.pyplot as plt

try:
    from cnn_model.constants import MIN_DEPTH, MAX_DEPTH
except:
    from constants import MIN_DEPTH, MAX_DEPTH

def DepthNorm(depth, maxDepth=MAX_DEPTH):
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))

def show_tensor_img(tensor_img):
    plt.imshow(tensor_img.permute(1, 2, 0))
    plt.show()


def find_files_in_path(path, query, MAX_DEPTH = 10):
    results =[]
    for root, dirs, files in os.walk(path, topdown=True):
        if root.count(os.sep) - path.count(os.sep) == MAX_DEPTH - 1:
            del dirs[:]
        for name in files:
            if query in name:
                results.append(os.path.join(root, name))
    return results

def copy_files(src,dest,query,depth = 2):
    a = find_files_in_path(src, query, depth)
    for img in a:
        file_new_name = basename(Path(img).parent)
        copyfile(img, str(Path(dest)/(file_new_name+".png")))

src = r"/home/roeematan/PycharmProjects/Shape-From-Shading/cnn_model/results/sfs2_small_pics_data_16052021_145154"
dest = r"/home/roeematan/PycharmProjects/Shape-From-Shading/cnn_model/results/sfs2_small_pics_data_16052021_145154/predict"

copy_files(src,dest,"epoch")