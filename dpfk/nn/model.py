import efficientnet_pytorch
import torch
from torch import nn
from torchvision import transforms

from dpfk.nn import model


class FrameModel(nn.Module):

    def __init__(self, backbone, conv_channels=256, n_layers=4, dropout=0.4):
        super().__init__()
        self.backbone = backbone
        in_channels = backbone._conv_head.out_channels
        conv_1d = []
        for _ in range(n_layers):
            conv_1d.append(
                nn.Sequential(
                    nn.Conv1d(in_channels,
                              conv_channels,
                              kernel_size=3,
                              padding=1),
                    nn.ReLU(),
                    nn.BatchNorm1d(conv_channels),
                ))
            in_channels = conv_channels
        self._conv_1d = nn.Sequential(*conv_1d)
        self._avg_pool = nn.AdaptiveAvgPool1d(1)
        self._dropout = nn.Dropout()
        self._head = nn.Linear(conv_channels, 1)

    def forward(self, x):
        b, f, *_ = x.shape
        x = x.reshape(b * f, *_)
        x = self.backbone.extract_features(x)
        x = self.backbone._avg_pooling(x)
        x = x.reshape(b, f, -1)
        x = x.permute(0, 2, 1)
        x = self._conv_1d(x)
        x = self._avg_pool(x)
        x = x.reshape(b, -1)
        x = self._dropout(x)
        x = self._head(x)
        return x


def get_model_normalize_from_config(config, pretrained=True):
    config_model = config['model']
    model_name = config_model['type']
    model_kwargs = config_model['kwargs']
    if pretrained:
        model = efficientnet_pytorch.model.EfficientNet.from_pretrained(
            model_name, **model_kwargs)
    else:
        num_classes = model_kwargs['num_classes']
        model = efficientnet_pytorch.model.EfficientNet.from_name(
            model_name, override_params={'num_classes': num_classes})

    if config_model['kwargs']['advprop']:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    return model, normalize


def get_framemodel_normalize_from_config(config):
    backbone, norm = get_model_normalize_from_config(config)
    return FrameModel(backbone, **config['model'].get('frame_kwargs', {})), norm
