import dataclasses
import glob
import json
import os
from concurrent import futures
from os import path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from . import loader


@dataclasses.dataclass
class Video:
    name: str
    path: str
    label: bool

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


def get_instances_from_folder(folder):
    with open(path.join(folder, 'metadata.json')) as f:
        instances = []
        meta = json.load(f)
        for k, v in meta.items():
            instances.append(
                Video(name=k,
                      path=path.join(folder, k),
                      label=(v['label'] == 'FAKE')))
    return instances


def grab_frames(vid_path: str, n: int = 16, rgb=False):
    capture = cv2.VideoCapture(vid_path)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = set(np.linspace(0, n_frames - 1, n).astype(int))
    for i in range(n_frames):
        _ = capture.grab()
        if i in sample:
            success, im = capture.retrieve()
            if success:
                if rgb:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                yield im
    capture.release()


def save_ims(video_path, out_dir, resolution, grab_frames_kwargs=None):
    grab_frames_kwargs = grab_frames_kwargs or {}
    frames = grab_frames(video_path, **grab_frames_kwargs)
    _, vid_name = path.split(video_path)
    im_root, _ = path.splitext(vid_name)

    for ix, f in enumerate(frames):
        f = cv2.resize(f, resolution, interpolation=cv2.INTER_AREA)
        out_pth = f"{path.join(out_dir, im_root)}_{ix}.jpg"
        cv2.imwrite(out_pth, f)


def write_frames(folders, destination_folder='data/', resolution=(1024, 576)):
    for folder in tqdm(folders, desc="Folders parsed"):
        instances = get_instances_from_folder(folder)
        _, folder_name = path.split(folder)
        out_dir = path.join(destination_folder, folder_name)
        os.mkdir(out_dir)

        fn = lambda i: save_ims(i.path, out_dir, resolution)
        with futures.ThreadPoolExecutor(24) as pool:
            tqdm(pool.map(fn, instances),
                 desc="Videos parsed",
                 total=len(instances))


def write_faces(detector, input_base, output_base):
    os.mkdir(output_base)
    input_folders = glob.glob(path.join(input_base, '*'))
    for input_folder in tqdm(input_folders):
        _, folder_stub = path.split(input_folder)
        output_folder = path.join(output_base, folder_stub)
        os.mkdir(output_folder)
        input_paths = glob.glob(path.join(input_folder, '*.mp4'))
        dataset = loader.VideoDataset(input_paths, n=8)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 collate_fn=lambda x: x,
                                                 batch_size=None,
                                                 num_workers=12)
        for ims, pth in tqdm(dataloader):
            _, file = path.split(pth)
            stub, _ = path.splitext(file)
            out_pths = [
                path.join(output_folder, f"{stub}_{ix}.jpg")
                for ix in range(len(ims))
            ]
            with torch.no_grad():
                detector(ims, save_path=out_pths)
