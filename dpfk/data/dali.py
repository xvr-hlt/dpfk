import collections
from os import path
from random import shuffle

import numpy as np

from nvidia.dali import ops, pipeline, types


class ImageIterator(object):

    def __init__(self, files, labels, batch_size, device_id, num_gpus):
        self.batch_size = batch_size
        world = len(files)
        self.labels = labels
        self.files = files[world * device_id // num_gpus:world *
                           (device_id + 1) // num_gpus]
        self.n = len(self.files)

    def __iter__(self):
        self.i = 0
        shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            raise StopIteration

        for _ in range(self.batch_size):
            file_path = self.files[self.i]
            _, file = path.split(file_path)
            stub = file.split('_')[0]
            label = self.labels[stub]
            with open(file_path, 'rb') as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))

            labels.append(np.array([label], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    @property
    def size(self):
        return self.n

    next = __next__


class ImagePipeline(pipeline.Pipeline):

    def __init__(self,
                 batch_size,
                 num_threads,
                 device_id,
                 external_data,
                 resize_x=1024,
                 resize_y=576):
        super().__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu",
                              resize_x=resize_x,
                              resize_y=resize_y)
        self.cmn = ops.CropMirrorNormalize(device="gpu",
                                           output_dtype=types.FLOAT,
                                           image_type=types.RGB,
                                           mean=[0., 0., 0.],
                                           std=[255., 255., 255.])
        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmn(images)
        return (output, self.labels)

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.jpegs, images)
            self.feed_input(self.labels, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


def get_image_pipeline(image_paths, labels, batch_size, device_id, num_gpus,
                       num_threads):
    image_iterator = ImageIterator(image_paths,
                                   labels=labels,
                                   batch_size=batch_size,
                                   device_id=device_id,
                                   num_gpus=num_gpus)
    return ImagePipeline(batch_size=batch_size,
                         num_threads=num_threads,
                         device_id=device_id,
                         external_data=image_iterator)
