import cv2
import torch
from torchvision import transforms

from dpfk import util
from dpfk.data import util as data_util
from dpfk.nn import model


class InferenceEngine:

    def __init__(self, config, weights, device='cuda'):
        config = util.read_config(config)
        self.model, normalize = model.get_model_normalize_from_config(
            config, pretrained=False)
        util.load_weights(self.model, weights)
        self.model = self.model.to(device)
        self.preprocess = transforms.Compose([transforms.ToTensor(), normalize])
        self.device = device

    @torch.no_grad()
    def __call__(self, vid_path):
        preds = []
        frames = data_util.grab_frames(vid_path)
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.preprocess(frame).to(self.device)
            frame = frame[None]
            preds.append(self.model(frame).sigmoid())
        return torch.cat(preds).mean()
