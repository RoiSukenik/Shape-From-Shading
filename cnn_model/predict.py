from PIL import Image
from io import BytesIO
import numpy as np
from os import listdir
from os.path import isfile, join
import torch
from os.path import basename
from pathlib import Path
import gc
try:
    from cnn_model.model import Model
    from cnn_model.utils import show_tensor_img, DepthNorm
    from cnn_model.pytorch_msssim import ssim
except:
    from model import Model
    from utils import show_tensor_img, DepthNorm
    from pytorch_msssim import ssim

import matplotlib

matplotlib.use('Agg')

SSIM_WEIGHT = 1.0
L1_WEIGHT = 0.1
HORZ_FLIP=False
HORZ_FLIP=False
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import torch.nn as nn

CUDA = torch.cuda.is_available()
try:
    with open("last_model.txt", "r") as text_file:
        MODEL_TO_LOAD = text_file.readline().strip()
except Exception as e:
    PATH = str(Path(__file__).parent.absolute()) + "/"
    with open(PATH + "last_model.txt", "r") as text_file:
        MODEL_TO_LOAD = text_file.readline().strip()

# MODEL_TO_LOAD = "sfs2_small_pics_data_12052021_222948_101_1.00e-06_50.pth"
MIDTRAIN_TEST_DIR = "/small_test_dir"

TEST_DIR = "/test_data/models/"
TEST_DIR = "/test_data/change_ls_amount/"
TEST_DIR = "/test_data/antispoffing"
TEST_DIR = "/test_data/phase_1_data"
TEST_DIR = "/test_data/mat_printed"
# TEST_DIR = "/test_data/phase_1_data_rgb"
TEST_DIR = "/small_test_dir"
TEST_DIR = "/test_data"
TEST_DIR = "/test_data/phase_1_data_rgb"

BIG_PICS = True if "small" not in MIDTRAIN_TEST_DIR else False
SAVE_CSV = True
GPU_TO_RUN = 2

import cv2
from skimage import exposure
import numpy as np


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


def load_image(im_path, is_depth=False):
    img = Image.open(im_path)
    img = img.convert('L')

    np_img = np.array(img, dtype=np.uint8)


    #np_img = match_image(np_img, im_path)

    np_img = np.dstack([np_img, np_img, np_img])
    image = Image.fromarray(np_img, 'RGB')

    if is_depth:
        image = image.resize((160, 120))
    else:
        if BIG_PICS:
            image = image.resize((320, 240))
    if "phase_1_data_rgb" in TEST_DIR:
        image = image.resize((320, 240))
    image = to_tensor(image)

    return image





def match_image(src, src_path):
    ref_dir = "/small_test_dir/00000_"
    ref_dir = str(Path(__file__).parent.absolute()) + ref_dir
    if "lu" in src_path:
        ref_path = ref_dir + "lu.png"
    if "rd" in src_path:
        ref_path = ref_dir + "rd.png"
    if "ru" in src_path:
        ref_path = ref_dir + "ru.png"
    if "ld" in src_path:
        ref_path = ref_dir + "ld.png"
    img = Image.open(ref_path)
    img = img.convert('L')
    ref= np.array(img, dtype=np.uint8)
    kernel = np.ones((5, 5), np.float32) / 25

    ref[ref < 210] += 40

    src = src[..., np.newaxis]
    ref = ref[..., np.newaxis]

    # determine if we are performing multichannel histogram matching
    # and then perform histogram matching itself
    print("[INFO] performing histogram matching...")
    multi = True if src.shape[-1] > 1 else False
    matched = exposure.match_histograms(src, ref, multichannel=multi)
    from matplotlib import pyplot as plt
    plt.subplot(121),plt.imshow(src,cmap = 'gray' ),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(matched,cmap = 'gray' ),plt.title('ref')
    plt.xticks([]), plt.yticks([])
    plt.show()


    return matched

def get_input_img(test_dir):
    data_dir = str(Path(__file__).parent.absolute()) + test_dir

    onlyimg = [join(data_dir, f) for f in listdir(data_dir) if isfile(join(data_dir, f))]
    return onlyimg


def plot_output_image_3D(depth_tensor, fname, output_path=None, params=None):
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
            plt.savefig(str(output_path.parent / "predict" / run_id))
    else:
        file_path = str(Path(__file__).parent.absolute()) + "/mid_train_results/" + fname.split('.')[0]

    if "depth" in fname:
        if SAVE_CSV:
            a = np.asarray(outputImageRealWorldScale)
            np.savetxt(str(Path(__file__).parent.absolute()) + "/mid_train_results/" + f"{fname}.csv", a, delimiter=",")

        # ax.view_init(elev=10., azim=180) # side view
        file_path = str(Path(__file__).parent.absolute()) + "/mid_train_results/" + f"{fname}.png"
        plt.savefig(file_path)

        ax.view_init(elev=10, azim=180)
        file_path = str(Path(__file__).parent.absolute()) + "/mid_train_results/" + f"{fname}_{180}.png"
        plt.savefig(file_path)
        plt.show()
    else:
        # ax.view_init(elev=10., azim=180) # side view
        plt.savefig(file_path)
    plt.close("all")


def show_net_output(output, fname=MODEL_TO_LOAD, output_path=None, params=None):
    # output =output / torch.max(output)
    # m = nn.Sigmoid()
    # output = m(output)
    plot_output_image_3D(output, fname, output_path, params)




def calc_loss(output, depth):
    depth_ns = DepthNorm(depth)

    depth_lst = []
    for dept in depth_ns:
        depth_n1 = dept[0]
        depth_n2 = depth_n1.unsqueeze(0)
        depth_lst.append(depth_n2)
    depth_n = torch.stack(depth_lst)

    l1_criterion = nn.L1Loss()

    background = depth_n == 1
    depth_masked = depth_n.masked_fill_(background, 0.0)
    output_masked = output.masked_fill_(background, 0.0)

    l_depth = l1_criterion(output_masked, depth_masked)
    l_ssim = 1 - ssim(output_masked, depth_masked, data_range=1, size_average=False,
                      nonnegative_ssim=True)  # (N,)

    loss = (SSIM_WEIGHT * l_ssim) + (L1_WEIGHT * l_depth)
    return loss.sum().item(), l_ssim.sum().item(), l_depth.sum().item()


def test_predict(model, epoch, test_loader, output_path=None, params=None):
    loss = 0
    ssim_loss = 0
    l1_loss = 0
    total = 0
    test_len = len(test_loader)
    calc_amount = test_len
    test_amount_iter = int(test_len/calc_amount)
    print_amount = 10
    total_showed = 0
    total_calc = 0
    for i, sample_batched in enumerate(test_loader):
        if i%test_amount_iter != 0:
            continue
        
        if CUDA:
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        else:
            image = torch.autograd.Variable(sample_batched['image'])
            depth = torch.autograd.Variable(sample_batched['depth'])

        output = model(image)
        if total_showed != print_amount and i%6 == 0:
            show_net_output(output, f"epoch_{epoch:02d}_{i}", output_path, params)
            total_showed+=1
        tot_loss, l_ssim, l_depth = calc_loss(output, depth)
        loss += tot_loss
        ssim_loss += l_ssim
        l1_loss += l_depth
        total_calc+=1
    test_loss = loss / (total_calc)
    l1_loss = l1_loss / (total_calc)
    ssim_loss = ssim_loss / (total_calc)

    return round(test_loss,3), round(l1_loss,3),round(ssim_loss,3)



def predict():
    if CUDA:
        model = Model().cuda()
        model = nn.DataParallel(model,device_ids=[GPU_TO_RUN])
        model.to(f'cuda:{model.device_ids[0]}')
    else:
        model = Model()
    im_order = ["lu", "ld", "ru", "rd"]
    imgs_path = get_input_img(TEST_DIR)
    sorted_imgs = []
    if HORZ_FLIP:
        im_order = ["ru", "rd", "lu", "ld"]
    im1 = [im_path for im_path in imgs_path if im_order[0] in im_path]
    im2 = [im_path for im_path in imgs_path if im_order[1] in im_path]
    im3 = [im_path for im_path in imgs_path if im_order[2] in im_path]
    im4 = [im_path for im_path in imgs_path if im_order[3] in im_path]

    imgs_path = im1 + im2 +im3 + im4

    images = [load_image(im_path) for im_path in imgs_path if "depth" not in im_path]
    depth = [load_image(im_path, is_depth=True) for im_path in imgs_path if "depth" in im_path]
    batch = torch.stack(images)
    if CUDA:
        batch = batch.cuda()

    model_PATH = MODEL_TO_LOAD

    show_tensor_img(images[3])

    model.load_state_dict(torch.load(model_PATH))
    model.eval()
    output = model(batch)
    show_net_output(output, "depth")


#
# def predict():
#     if CUDA:
#         model = Model().cuda()
#         model = nn.DataParallel(model, device_ids=[0])
#         model.to(f'cuda:{model.device_ids[0]}')
#     else:
#         model = Model()
#     model_PATH = MODEL_TO_LOAD
#     model.load_state_dict(torch.load(model_PATH))
#     model.eval()
#
#     # show_tensor_img(images[3])
#     losss = []
#
#     for i in range(4):
#         data_path = TEST_DIR + f"{i + 1}"
#         imgs_path = get_input_img(data_path)
#
#         im1 = [im_path for im_path in imgs_path if "lu" in im_path]
#         im2 = [im_path for im_path in imgs_path if "ld" in im_path]
#         im3 = [im_path for im_path in imgs_path if "ru" in im_path]
#         im4 = [im_path for im_path in imgs_path if "rd" in im_path]
#         imgs_path = im1 + im2 + im3 + im4
#         print([basename(Path(img_path)).split('_')[1] for img_path in imgs_path])
#         images = [load_image(im_path) for im_path in imgs_path if "depth" not in im_path]
#         batch = torch.stack(images)
#
#         if CUDA:
#             batch = batch.cuda()
#
#         output = model(batch)
#         show_net_output(output, f"depth{i}")
#
#         imgs_path = get_input_img(data_path)
#         depth = [load_image(im_path, is_depth=True) for im_path in imgs_path if "depth" in im_path]
#         depth_n = depth[0]
#         depth_n = depth_n.unsqueeze(0)
#         depth_n = depth_n[0][0]
#         depth_n = torch.reshape(depth_n, (1, 1, depth_n.shape[0], depth_n.shape[1])).cuda()
#
#         losss.append(calc_loss(depth_n, output).item())
#
#     print(losss)
#     plt.plot([1, 2, 3, 4], losss)
#     plt.show()
#     print(3)
#     #
#     # plot_output_image_3D(depth_n, "depth")


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()
    with torch.cuda.device(GPU_TO_RUN):
        predict()
