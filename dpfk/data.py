import dataclasses
import json
from os import path
from typing import Optional

import cv2


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


def grab_frames(vid_path: str, nth: int = 10):
    capture = cv2.VideoCapture(vid_path)
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(n_frames):
        _ = capture.grab()
        if i % nth == 0:
            _, im = capture.retrieve()
            yield im


def save_ims(video_path, out_dir, resolution, grab_frames_kwargs=None):
    grab_frames_kwargs = grab_frames_kwargs or {}
    frames = grab_frames(video_path, **grab_frames_kwargs)
    _, vid_name = path.split(video_path)
    im_root, _ = path.splitext(vid_name)

    for ix, f in enumerate(frames):
        f = cv2.resize(f, resolution, interpolation=cv2.INTER_AREA)
        out_pth = f"{path.join(out_dir, im_root)}_{ix}.jpg"
        cv2.imwrite(out_pth, f)
