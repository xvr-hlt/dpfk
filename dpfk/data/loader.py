import glob
from os import path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import dpfk


class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, files, labels, normalization):
        self.files = files
        self.labels = labels
        self.normalization = transforms.Compose(
            [transforms.ToTensor(), normalization])

    def __getitem__(self, i):
        file_path = self.files[i]
        im = Image.open(file_path)
        im = self.normalization(im)

        _, f = path.split(file_path)
        stub = f.split('_')[0]
        label = self.labels[stub]
        label = torch.Tensor([label]).float()

        return im, label

    def __len__(self):
        return len(self.files)

    @classmethod
    def get_train_loader(cls, config, normalize):
        data_conf = config['data']
        _, *train_folders = sorted(
            glob.glob(path.join(data_conf['instances_home'], '*')))
        train_instances = [
            i for f in train_folders
            for i in dpfk.data.util.get_instances_from_folder(f)
        ]
        labels = {i.name.replace('.mp4', ''): i.label for i in train_instances}
        _, *train_image_folders = sorted(
            glob.glob(path.join(data_conf['images_home'], '*')))
        train_images = [
            im for f in train_image_folders
            for im in glob.glob(path.join(f, '*.jpg'))
        ]
        return cls(train_images, labels, normalize)

    @classmethod
    def get_val_loader(cls, config, normalize):
        data_conf = config['data']
        val_folder, *_ = sorted(
            glob.glob(path.join(data_conf['instances_home'], '*')))
        val_instances = dpfk.data.util.get_instances_from_folder(val_folder)
        labels = {i.name.replace('.mp4', ''): i.label for i in val_instances}
        val_images = glob.glob(path.join(data_conf['images_home'], '*_0/*.jpg'))
        return cls(val_images, labels, normalize)
