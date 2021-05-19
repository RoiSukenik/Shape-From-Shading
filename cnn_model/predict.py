from PIL import Image
from io import BytesIO
import numpy as np
from os import listdir
from os.path import isfile, join
import torch
from os.path import basename
from pathlib import Path

try:
    from cnn_model.model import Model
    from cnn_model.utils import show_tensor_img
except:
    from model import Model
    from utils import show_tensor_img
import matplotlib
matplotlib.use('Agg')


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch.nn as nn
CUDA = torch.cuda.is_available()
try:
    with open("last_model.txt", "r") as text_file:
        MODEL_TO_LOAD = text_file.readline().strip()
except Exception as e:
    PATH = str(Path(__file__).parent.absolute()) +"/"
    with open(PATH + "last_model.txt", "r") as text_file:
        MODEL_TO_LOAD = text_file.readline().strip()


#MODEL_TO_LOAD = "sfs2_small_pics_data_12052021_222948_101_1.00e-06_50.pth"
TEST_DIR = "/small_test_dir"
BIG_PICS = True if "small" not in TEST_DIR else False
SAVE_CSV = True

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError(
            'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))

        return img.float().div(255)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(
            torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)

    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def load_image(im_path, is_depth = False):
    img = Image.open(im_path)
    img = img.convert('L')
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img, np_img, np_img])
    image = Image.fromarray(np_img, 'RGB')

    if is_depth:
        image = image.resize((160, 120))
    else:
        if BIG_PICS:
          image = image.resize((320, 240))
    image = to_tensor(image)

    return image

def get_input_img():
    data_dir = str(Path(__file__).parent.absolute()) + TEST_DIR

    onlyimg = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
    return onlyimg

def plot_output_image_3D(depth_tensor, fname, output_path = None, params = None):
    depth_width = 120
    depth_height = 160
    outputImageRealWorldScale = depth_tensor.detach().cpu().numpy().reshape(depth_width, depth_height)
    outputImageRealWorldScale = (20 * depth_tensor) + 10
    outputImageRealWorldScale = outputImageRealWorldScale.detach().cpu().numpy().reshape(depth_width, depth_height)


    xx = range(depth_height)
    yy = range(depth_width)
    X, Y = np.meshgrid(xx, yy)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if params:
        tit = f'lr: {params["LEARNING_RATE"]}, ssim: {params["SSIM_WEIGHT"]}, l1: {params["L1_WEIGHT"]}, Accumulation: {params["ACCUMULATION_STEPS"]}\n'
        tit += f'Scheduler: {params["USE_SCHEDULER"]}, Step Size: {params["SCHEDULER_STEP_SIZE"]}, Gamma: {params["SCHEDULER_GAMMA"]}, '
        tit += f'Adaptive: {params["ADAPTIVE_LEARNER"]}'
    else:
        tit = fname.split('.')[0]
    ax.set_title(tit)
    ax.plot_surface(X, Y, outputImageRealWorldScale)
    if output_path:
        file_path = str(output_path / fname.split('.')[0])
        if "mid_run" not in str(output_path):
            if SAVE_CSV:
                a = np.asarray(outputImageRealWorldScale)
                np.savetxt(file_path + ".csv", a, delimiter=",")
            run_id = str(basename(output_path))
            plt.savefig(str(output_path.parent/ "predict" / run_id))
    else:
        file_path = str(Path(__file__).parent.absolute()) + "/mid_train_results/" + fname.split('.')[0]

    if fname == "depth":
        if SAVE_CSV:
            a = np.asarray(outputImageRealWorldScale)
            np.savetxt(str(Path(__file__).parent.absolute()) + "/mid_train_results/" +  f"depth.csv", a, delimiter=",")
        
        # ax.view_init(elev=10., azim=180) # side view
        file_path = str(Path(__file__).parent.absolute()) + "/mid_train_results/" +  f"depth.png"
        plt.savefig(file_path)
        
        ax.view_init(elev=10, azim=180)
        file_path = str(Path(__file__).parent.absolute()) + "/mid_train_results/" +  f"depth{180}.png"
        plt.savefig(file_path)
        #plt.show()
    else:
        # ax.view_init(elev=10., azim=180) # side view
        plt.savefig(file_path)
    plt.close("all")

def show_net_output(output, fname = MODEL_TO_LOAD, output_path = None, params = None):
    # output =output / torch.max(output)
    #m = nn.Sigmoid()
    #output = m(output)
    plot_output_image_3D(output, fname, output_path, params)


def test_predict(model, epoch, output_path = None, params = None):
    imgs_path = get_input_img()
    sorted_imgs = []
    im1 = [im_path for im_path in imgs_path if "lu" in im_path]
    im2 = [im_path for im_path in imgs_path if "ld" in im_path]
    im3 = [im_path for im_path in imgs_path if "ru" in im_path]
    im4 = [im_path for im_path in imgs_path if "rd" in im_path]
    imgs_path = im1 + im2 +im3 + im4

    images = [load_image(im_path) for im_path in imgs_path if "depth" not in im_path]
    depth = [load_image(im_path, is_depth=True) for im_path in imgs_path if "depth" in im_path]
    batch = torch.stack(images)
    if CUDA:
        batch = batch.cuda()
        
    output = model(batch)
    show_net_output(output, f"epoch_{epoch:02d}", output_path, params)
     
     
def predict():
    if CUDA:
        model = Model().cuda()
        model = nn.DataParallel(model,device_ids=[3])
        model.to(f'cuda:{model.device_ids[0]}')
    else:
        model = Model()

    imgs_path = get_input_img()
    sorted_imgs = []
    im1 = [im_path for im_path in imgs_path if "lu" in im_path]
    im2 = [im_path for im_path in imgs_path if "ld" in im_path]
    im3 = [im_path for im_path in imgs_path if "ru" in im_path]
    im4 = [im_path for im_path in imgs_path if "rd" in im_path]
    imgs_path = im1 + im2 +im3 + im4

    images = [load_image(im_path) for im_path in imgs_path if "depth" not in im_path]
    depth = [load_image(im_path, is_depth=True) for im_path in imgs_path if "depth" in im_path]
    batch = torch.stack(images)
    if CUDA:
        batch = batch.cuda()

    model_PATH = MODEL_TO_LOAD

    # show_tensor_img(images[3])

    model.load_state_dict(torch.load(model_PATH))
    model.eval()
    output = model(batch)
    show_net_output(output, "depth")


    # depth_n = depth[0]
    # depth_n = depth_n.unsqueeze(0)
    # depth_n = depth_n[0][0]
    # depth_n = torch.reshape(depth_n, (1, 1, depth_n.shape[0], depth_n.shape[1]))
    #
    # plot_output_image_3D(depth_n, "depth")


if __name__ == '__main__':
    with torch.cuda.device(3):
        predict()
