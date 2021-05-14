import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random
import os
import shutil
from os import listdir
from os.path import isfile, join
from pathlib import Path

try:
    from cnn_model.constants import MIN_DEPTH, MAX_DEPTH, DATA_NAME, DATA_PATH

except:
    from constants import MIN_DEPTH, MAX_DEPTH, DATA_NAME, DATA_PATH

def create_pdf(input_path,to_zip=True):
    test_dir = "sfs_test"
    train_dir = "sfs_train"

    imgs = ["lu.png", "ld.png","ru.png","rd.png",]
    train_data_ims = []
    train_data_depth = []
    test_data_ims = []
    test_data_depth = []
    for root, dirs, files in os.walk(input_path):
        for name in files:
            if "depth" in name:
                depth_img = name
                im_num = name.split('_')[0]
                im_names = [im_num +'_'+ im_name for im_name in imgs]
                im_names = [im_name.replace('\\','/')for im_name in im_names]
                depth_img=depth_img.replace('\\','/')
                if test_dir in root:
                    for im_name in im_names:
                        test_data_ims.append(os.path.join(root[root.index("data"):], im_name))
                        test_data_depth.append(os.path.join(root[root.index("data"):], depth_img))

                if train_dir in root:
                    for im_name in im_names:
                        train_data_ims.append(os.path.join(root[root.index("data"):], im_name))
                        train_data_depth.append(os.path.join(root[root.index("data"):], depth_img))

    test_pdf ={"image_path": test_data_ims, "depth_path" : test_data_depth}
    train_pdf ={"image_path": train_data_ims, "depth_path" : train_data_depth}

    df_test = pd.DataFrame(test_pdf, columns=['image_path', 'depth_path'])  # create DataFrame
    df_train = pd.DataFrame(train_pdf, columns=['image_path', 'depth_path'])  # create DataFrame

    df_test.to_csv(input_path+'\\sfs_test.csv', header=False, index=False)
    df_train.to_csv(input_path+'\\sfs_train.csv', header=False, index=False)

    if to_zip:
        shutil.make_archive("sfs2_data", 'zip', base_dir="data")

    print("Files saved successfully")

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}


def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list(
        (row.split(',') for row in (data['data/' + DATA_NAME + '_train.csv']).decode("utf-8").split('\n') if len(row) > 0))

    from sklearn.utils import shuffle
    # nyu2_train = shuffle(nyu2_train, random_state=0)

    # if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.nyu_dataset[idx]
        sample[0] = sample[0].replace('\\','/')
        sample[1] = sample[1].replace('\\','/')
        sample[1] = sample[1].replace('\r','')
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))

        img = image.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        np_img = Image.fromarray(np_img, 'RGB')

        dep = depth.convert('L')
        np_dep = np.array(dep, dtype=np.uint8)
        np_dep = np.dstack([np_dep, np_dep, np_dep])
        np_dep = Image.fromarray(np_dep, 'RGB')

        sample = {'image': np_img, 'depth': np_dep}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        ### already resized pics
        if image.shape[0] != 320:
            image = image.resize((320, 240))

        image = self.to_tensor(image)

        depth = depth.resize((160, 120))
        if self.is_test:
            depth = self.to_tensor(depth).float() /int(MAX_DEPTH)
        else:
            depth = self.to_tensor(depth).float() *int(MAX_DEPTH)

        # put in expected range
        depth = torch.clamp(depth, int(MIN_DEPTH),int(MAX_DEPTH))

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
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


def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])


def getDefaultTrainTransform():
    return transforms.Compose([
        ToTensor()
    ])

def change_path():
    dir_path  = Path(DATA_PATH).parent.absolute()
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    print("\nChoose new data to load: ")
    print([f"{i}: {data}" for i,data in enumerate(onlyfiles)])
    choice = input("Your choice: ")
    return str(dir_path)+'/'+onlyfiles[int(choice)]

def getTrainingTestingData(batch_size):
    path_to_load = DATA_PATH
    print(f"Trying to load: {path_to_load}")
    dir_path  = Path(DATA_PATH).parent.absolute()
    onlyfiles = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    print("Data available: ", onlyfiles)
    try:
        data, nyu2_train = loadZipToMem(path_to_load)
    except FileNotFoundError:
        path_to_load = change_path()
        data, nyu2_train = loadZipToMem(path_to_load)

    print(f"Loaded {path_to_load}")

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=False), DataLoader(transformed_testing, batch_size,
                                                                                  shuffle=False)
if __name__ == '__main__':
    create_pdf("C:\\Users\\machiel\\PycharmProjects\\Shape-From-Shading\\cnn_model\\data")