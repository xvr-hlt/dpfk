import glob
from os import path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import dpfk


class ImageLoader(torch.utils.data.Dataset):

    @staticmethod
    def is_validation_fold(fold):
        suffix = fold.split('_')[-1]
        return suffix in {'0', '18', '27', '36', '45'}

    def __init__(self, files, labels, normalization, size, augmentation=None):
        if isinstance(size, list):
            size = tuple(size)
        self.files = files
        self.labels = labels
        pipeline = []
        if augmentation is not None:
            pipeline.append(augmentation)
        pipeline.append(transforms.Resize(size))
        pipeline.append(transforms.ToTensor())
        pipeline.append(normalization)
        self.pipeline = transforms.Compose(pipeline)

    def __getitem__(self, i):
        file_path = self.files[i]
        im = Image.open(file_path)
        im = self.pipeline(im)

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
            i for f in train_folders
            for i in dpfk.data.util.get_instances_from_folder(f)
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
            i for f in val_folders
            for i in dpfk.data.util.get_instances_from_folder(f)
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
        augs.append(transforms.RandomHorizontalFlip(prob))
        if aug_level > 0:
            distortion_scale = [None, 0.125, 0.25, 0.5, 0.75][aug_level]
            augs.append(
                transforms.RandomPerspective(distortion_scale=distortion_scale,
                                             p=prob))

            degrees = [None, 20., 40., 60., 80.][aug_level]
            augs.append(
                transforms.RandomApply([transforms.RandomRotation(degrees)],
                                       p=prob))
            scale = [None, (0.9, 1.0), (0.8, 1.0), (0.7, 1.0),
                     (0.8, 1.0)][aug_level]
            ratio = [None, (0.9, 1.1), (0.8, 1.2), (0.7, 1.3),
                     (0.8, 1.4)][aug_level]

            augs.append(
                transforms.RandomApply([
                    transforms.RandomResizedCrop(
                        size=size, scale=scale, ratio=ratio)
                ],
                                       p=prob))
        return transforms.Compose(augs)
