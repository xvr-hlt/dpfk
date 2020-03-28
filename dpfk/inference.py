import glob
from os import path

import facenet_pytorch
import torch
import yaml

from dpfk import util
from dpfk.data import util as data_util
from dpfk.nn import model


class InferenceEngine:

    def __init__(self, run_dir, n=12, device='cuda'):
        self.run_dir = run_dir
        with open(path.join(run_dir, 'infer.yml')) as f:
            self.config = yaml.safe_load(f)
        self.facedet = facenet_pytorch.MTCNN(device=device,
                                             **self.config['facedet_kwargs'])
        self.device = device
        self._model = None
        self.n = n

        model_dir = max(glob.glob(path.join(run_dir, 'runs', '*')))
        config = path.join(model_dir, 'config.yaml')
        model_pth = max(glob.glob(path.join(model_dir, '_ckpt_epoch_*.ckpt')))
        model_config = util.read_config(config)
        self.model, self.normalize = model.get_model_normalize_from_config(
            model_config, pretrained=False)
        util.load_weights(self.model, model_pth)
        self.model = self.model.to(device)

    @torch.no_grad()
    def __call__(self, vid_path):
        frames = list(data_util.grab_frames(vid_path, n=self.n, rgb=True))
        faces = self.facedet(frames)
        faces = [self.normalize(f / 255.) for f in faces if f is not None]
        faces = torch.stack(faces)
        faces = faces.to(self.device)
        probs = self.model(faces)
        return probs.sigmoid().mean()
