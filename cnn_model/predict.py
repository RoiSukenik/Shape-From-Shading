from PIL import Image
from io import BytesIO
import numpy as np
from os import listdir
from os.path import isfile, join
import torch
import pathlib
from cnn_model.model import Model
from cnn_model.utils import show_tensor_img
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch.nn as nn
PATH = str(pathlib.Path(__file__).parent.absolute()) + "\\saved_model"
CUDA = torch.cuda.is_available()
MODEL_TO_LOAD = "182612_3672_1.0000000000000002e-06_20.pth"


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


def load_image(im_path):
    img = Image.open(im_path)
    img = img.convert('L')
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img, np_img, np_img])
    image = Image.fromarray(np_img, 'RGB')

    image = image.resize((320, 240))  # trial
    image = to_tensor(image)

    return image

def get_input_img():
    data_dir = str(pathlib.Path(__file__).parent.absolute()) + "/test_dir"

    onlyimg = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
    return onlyimg

def plot_output_image_3D(depth_tensor):

    outputImageRealWorldScale = depth_tensor.detach().cpu().numpy().reshape(120, 160)
    outputImageRealWorldScale = (20 * depth_tensor) + 10
    outputImageRealWorldScale = outputImageRealWorldScale.detach().cpu().numpy().reshape(120, 160)
    xx = range(160)
    yy = range(120)
    X, Y = np.meshgrid(xx, yy)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, outputImageRealWorldScale)
    plt.show()


def predict():
    if CUDA:
        model = Model().cuda()
    else:
        model = Model()

    imgs_path = get_input_img()

    images = [load_image(im_path) for im_path in imgs_path if "depth" not in im_path]
    batch = torch.stack(images)
    if CUDA:
        batch = batch.cuda()

    model_PATH = str(pathlib.Path(__file__).parent.absolute()) + "/saved_model/" + MODEL_TO_LOAD

    # show_tensor_img(images[3])

    model.load_state_dict(torch.load(model_PATH))
    model.eval()
    output = model(batch)

    output =output / torch.max(output)

    m = nn.Sigmoid()
    output = m(output)
    plot_output_image_3D(output)


if __name__ == '__main__':
    predict()
