import os
import re
import struct
import glob
import numpy as np
import skimage
import skimage.io
from core.utils.utils import InputPadder

from PIL import Image

import torch
from torch.utils.data import Dataset


class KLens(Dataset):
    def __init__(self, root_path="",split="valid",ref="",meas=""):
        super(KLens, self).__init__()
        self.split = split
        file_list = {}
        file_list['train'] = []
        file_list['valid'] = []
        file_list['test'] = []
        file_list['train+valid'] = []
        for middle in glob.glob("/data2/opticalflow/KLENS/videos/1/frame_*_2.jpg"):
            file_list["train"].append([middle,middle[:-5]+"0.jpg"])
            file_list["valid"].append([middle,middle[:-5]+"0.jpg"])
            file_list["test"].append([middle,middle[:-5]+"0.jpg"])
            file_list["train+valid"].append([middle,middle[:-5]+"0.jpg"])
            file_list["train"].append([middle,middle[:-5]+"1.jpg"])
            file_list["valid"].append([middle,middle[:-5]+"1.jpg"])
            file_list["test"].append([middle,middle[:-5]+"1.jpg"])
            file_list["train+valid"].append([middle,middle[:-5]+"1.jpg"])
            file_list["train"].append([middle,middle[:-5]+"3.jpg"])
            file_list["valid"].append([middle,middle[:-5]+"3.jpg"])
            file_list["test"].append([middle,middle[:-5]+"3.jpg"])
            file_list["train+valid"].append([middle,middle[:-5]+"3.jpg"])
            file_list["train"].append([middle,middle[:-5]+"4.jpg"])
            file_list["valid"].append([middle,middle[:-5]+"4.jpg"])
            file_list["test"].append([middle,middle[:-5]+"4.jpg"])
            file_list["train+valid"].append([middle,middle[:-5]+"4.jpg"])

        self.dataset = file_list

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx):
        im0_path, im1_path = self.dataset[self.split][idx]

        images= self.load_image_list([im0_path,im1_path])
        # print(images[])
        for i in range(images.shape[0]-1):
            image0 = images[i,None]
            image1 = images[i+1,None]
        # img0 = skimage.io.imread(im0_path)
        # img1 = skimage.io.imread(im1_path)
        # img0 = torch.tensor(img0/255.).float()
        # img1 = torch.tensor(img1/255.).float()

        return image0, image1, im0_path , im1_path
    
    def load_image(self,imfile):
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img
    
    def load_image_list(self,image_files):
        images = []
        for imfile in image_files:
            images.append(self.load_image(imfile))
    
        images = torch.stack(images, dim=0)
        # images = images.to("cuda")

        padder = InputPadder(images.shape)
        return padder.pad(images)[0]

class Flo:
    def __init__(self, w, h):
        self.__floec1__ = float(202021.25)
        self.__floec2__ = int(w)
        self.__floec3__ = int(h)
        self.__floheader__ = struct.pack('fii', self.__floec1__, self.__floec2__, self.__floec3__)
        self.__floheaderlen__ = len(self.__floheader__)
        self.__flow__ = w
        self.__floh__ = h
        self.__floshape__ = [self.__floh__, self.__flow__, 2]

        if self.__floheader__[:4] != b'PIEH':
            raise Exception('Expect machine to be LE.')

    def load(self, file):
        with open(file, 'rb') as fp:
            if fp.read(self.__floheaderlen__) != self.__floheader__:
                raise Exception('Bad flow header: ' + file)
            result = np.ndarray(shape=self.__floshape__,
                                dtype=np.float32,
                                buffer=fp.read(),
                                order='C')
            return result

    def save(self, arr, fname):
        with open(fname, 'wb') as fp:
            fp.write(self.__floheader__)
            fp.write(arr.astype(np.float32).tobytes())


