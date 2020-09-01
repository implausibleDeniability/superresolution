import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import crop as torch_crop

from readings import read_dcm, openImage


class Loader(torch.utils.data.Dataset):
    def __init__(self, type, amount=None, dataset="portable",
                 make_crop=True, crop_size=48):
        self.make_crop = make_crop
        self.crop_size = crop_size
        self.amount = amount
        if (type == 'val'):
            portable = open(f"directories/{dataset}_test.txt")
        else:
            portable = open(f"directories/{dataset}_train.txt")
        self.images = []
        for filename in portable:
            self.images.append(filename[:-1])
        self.transform = transforms.ToTensor()

    def __crop(self, imageLR, imageHR, crop_size):
        top = int(np.random.random() * (imageLR.size[1] - crop_size))
        left = int(np.random.random() * (imageLR.size[0] - crop_size))
        return (torch_crop(imageLR, top, left, crop_size, crop_size),
                torch_crop(imageHR, 2 * top, 2 * left,
                           2 * crop_size, 2 * crop_size))

    def __getitem__(self, idx):
        imageHR = openImage(self.images[idx])
        imageLR = imageHR.copy()
        imageLR = imageLR.resize((imageHR.size[0] // 2, imageHR.size[1] // 2),
                                 Image.NEAREST)
        if (self.make_crop):
            imageLR, imageHR = self.__crop(imageLR, imageHR, self.crop_size)
        imageHR = self.transform(imageHR)
        imageLR = self.transform(imageLR)
        return imageHR, imageLR

    def __len__(self):
        if self.amount is None:
            return len(self.images)
        else:
            return self.amount


class MultiscaleLoader(torch.utils.data.Dataset):
    def __init__(self, type, scale, amount=None, make_crop=True, crop_size=48):
        self.make_crop = make_crop
        self.crop_size = crop_size
        self.amount = amount
        self.scale = scale
        if (type == 'val'):
            portable = open(f"directories/testx{scale}.txt")
        else:
            portable = open(f"directories/trainx{scale}.txt")
        self.images = []
        for filename in portable:
            self.images.append(filename[:-1])
        self.transform = transforms.ToTensor()

    def __crop(self, imageLR, imageHR, crop_size):
        top = int(np.random.random() * (imageLR.size[1] - crop_size))
        left = int(np.random.random() * (imageLR.size[0] - crop_size))
        return (torch_crop(imageLR, top, left, crop_size, crop_size),
                torch_crop(imageHR, self.scale * top, self.scale * left,
                           self.scale * crop_size, self.scale * crop_size))

    def __getitem__(self, idx):
        imageHR = openImage(self.images[idx])
        imageHRcrop = np.array(imageHR.size) // self.scale * self.scale
        imageHR = imageHR.crop([0, 0] + list(imageHRcrop))
        imageLR = imageHR.copy()
        imageLR = imageLR.resize((imageHR.size[0] // self.scale,
                                  imageHR.size[1] // self.scale),
                                 Image.NEAREST)
        if (self.make_crop):
            imageLR, imageHR = self.__crop(imageLR, imageHR, self.crop_size)
        if (np.random.random() > 0.5):
            imageLR = imageLR.transpose(Image.FLIP_LEFT_RIGHT)
            imageHR = imageHR.transpose(Image.FLIP_LEFT_RIGHT)
        imageHR = self.transform(imageHR)
        imageLR = self.transform(imageLR)
        return imageHR, imageLR

    def __len__(self):
        if self.amount is None:
            return len(self.images)
        else:
            return max(self.amount, len(self.images))
