import efficientnet_pytorch
from torchvision import transforms


def get_model_normalize_from_config(config):
    config_model = config['model']
    model = efficientnet_pytorch.model.EfficientNet.from_pretrained(
        config_model['type'], **config_model['kwargs'])
    if config_model['kwargs']['advprop']:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    return model, normalize
