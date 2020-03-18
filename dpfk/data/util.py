import dataclasses
import json
import os
from concurrent import futures
from os import path
from typing import Optional

import cv2
import numpy as np
from tqdm.auto import tqdm


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


def grab_frames(vid_path: str, n: int = 32):
    capture = cv2.VideoCapture(vid_path)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = set(np.linspace(0, n_frames - 1, n).astype(int))
    for i in range(n_frames):
        _ = capture.grab()
        if i in sample:
            success, im = capture.retrieve()
            if success:
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


def write_frames(folders, destination_folder='data/', resolution=(576, 1024)):
    for folder in tqdm(folders, desc="Folders parsed"):
        instances = get_instances_from_folder(folder)
        _, folder_name = path.split(folder)
        out_dir = path.join(destination_folder, folder_name)
        os.mkdir(out_dir)

        fn = lambda i: save_ims(i.path, out_dir, resolution)
        with futures.ThreadPoolExecutor(12) as pool:
            tqdm(pool.map(fn, instances),
                 desc="Videos parsed",
                 total=len(instances))


"""
def scratch():
    mt = pd.read_csv('metadata.csv', index_col=0)
    audio_altered = ((mt['original'] != 'NAN') & (mt['wav.hash'] != mt['wav.hash.orig']))
    mt_audio = mt[audio_altered]
    g45 = mt_audio.folder.map(lambda x: int(x.split('_')[-1]) >= 45)   
    mt_audio[mt_audio['audio.@codec_time_base'] == '1/16000']
"""
