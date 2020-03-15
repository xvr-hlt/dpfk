import efficientnet_pytorch
from torchvision import transforms


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
