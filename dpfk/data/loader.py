import collections
import glob
import random
from os import path

import albumentations
import cv2
import numpy as np
import torch
from albumentations import pytorch as albumtorch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as FT

import dpfk

from . import util


class ImageLoader(torch.utils.data.Dataset):

    @staticmethod
    def is_validation_fold(fold):
        suffix = fold.split('_')[-1]
        return suffix in {'0', '18', '27', '36', '45'}

    def __init__(self, files, labels, normalization, size, augmentation=None):
        if isinstance(size, list):
            size = tuple(size)
        self.size = size
        self.files = files
        self.labels = labels
        pipeline = []
        if augmentation is not None:
            pipeline.append(augmentation)
        pipeline.append(albumentations.Resize(*size))
        pipeline.append(albumtorch.ToTensor())
        self.pipeline = albumentations.Compose(pipeline)
        self.normalization = normalization

    def __getitem__(self, i):
        file_path = self.files[i]
        im = Image.open(file_path)
        im = np.array(im)
        im = self.pipeline(image=im)['image']
        if self.normalization:
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
        folders = sorted(glob.glob(path.join(data_conf['instances_home'], '*')))
        train_folders = [f for f in folders if not cls.is_validation_fold(f)]
        train_instances = [
            i for f in train_folders for i in util.get_instances_from_folder(f)
        ]
        labels = {i.name.replace('.mp4', ''): i.label for i in train_instances}
        image_folders = sorted(
            glob.glob(path.join(data_conf['images_home'], '*')))
        train_image_folders = [
            f for f in image_folders if not cls.is_validation_fold(f)
        ]
        train_images = [
            im for f in train_image_folders
            for im in glob.glob(path.join(f, '*.jpg'))
        ]
        size = data_conf['size']
        aug_conf = config['aug']
        augmentation = cls.get_augmentation(aug_conf['prob'], aug_conf['level'],
                                            size)
        return cls(train_images, labels, normalize, size, augmentation)

    @classmethod
    def get_val_loader(cls, config, normalize):
        data_conf = config['data']
        folders = sorted(glob.glob(path.join(data_conf['instances_home'], '*')))
        val_folders = [f for f in folders if cls.is_validation_fold(f)]
        val_instances = [
            i for f in val_folders for i in util.get_instances_from_folder(f)
        ]
        labels = {i.name.replace('.mp4', ''): i.label for i in val_instances}
        image_folders = sorted(
            glob.glob(path.join(data_conf['images_home'], '*')))
        val_image_folders = [
            f for f in image_folders if cls.is_validation_fold(f)
        ]
        val_images = [
            im for f in val_image_folders
            for im in glob.glob(path.join(f, '*.jpg'))
        ]
        size = data_conf['size']
        return cls(val_images, labels, normalize, size)

    @staticmethod
    def get_augmentation(prob, aug_level, size):
        if isinstance(size, list):
            size = tuple(size)
        augs = []
        augs.append(albumentations.HorizontalFlip(prob))

        if aug_level > 0:
            shift_limit = [None, 0.0625, 0.125, 0.1875][aug_level]
            rotate_limit = [None, 20., 40., 60.][aug_level]
            scale_limit = [None, 0.1, 0.2, 0.3][aug_level]
            augs.append(
                albumentations.ShiftScaleRotate(
                    shift_limit,
                    scale_limit,
                    rotate_limit,
                    p=prob,
                    border_mode=cv2.BORDER_CONSTANT))
            augs.append(albumentations.Downscale(p=prob / 4))
        return albumentations.Compose(augs)


class FrameLoader(ImageLoader):
    N_FRAMES = 8

    def __init__(self, files, labels, normalization, size, augmentation=None):
        augmentation = None
        super().__init__(files, labels, normalization, size, augmentation)
        seqs = collections.defaultdict(list)

        for f in self.files:
            _, file = path.split(f)
            stub, _ = file.split('_')
            seqs[stub].append(f)

        self.seqs = [(k, sorted(v)) for k, v in seqs.items()]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        stub, paths = self.seqs[i]
        ims = torch.zeros((self.N_FRAMES, 3, *self.size))
        flip = random.random() > 0.5
        lq = random.random() > 0.1
        for ix, pth in enumerate(paths):
            im = Image.open(pth)
            im = np.array(im)
            if flip:
                im = albumentations.augmentations.functional.hflip(im)
            if lq:
                im = albumentations.augmentations.functional.downscale(
                    im, scale=0.25)
            im = self.pipeline(image=im)['image']
            if self.normalization:
                im = self.normalization(im)
            ims[ix] = im
        label = self.labels[stub]
        label = torch.Tensor([label]).float()
        return ims, label


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, paths, n=16):
        self.paths = paths
        self.n = n

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        pth = self.paths[ix]
        ims = list(util.grab_frames(pth, n=self.n, rgb=True))
        return ims, pth
