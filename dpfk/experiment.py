import glob
from os import path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.logging import WandbLogger
from torch import nn, optim
from torch.nn import functional as F

import dpfk.nn.model
from dpfk.data import loader, util


class Experiment(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model, self.normalize = dpfk.nn.model.get_model_normalize_from_config(
            config)
        self.loss = self.configure_loss()

        self.distributed_sampler = (
            config['trainer'].get('distributed_backend') == 'ddp')

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': self.loss(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        y_pred = y_hat.argmax(dim=1).bool()
        y_true = y.bool()

        tp = y_true[y_pred].sum()
        fp = y_true[~y_pred].sum()
        tn = (~y_true[~y_pred]).sum()
        fn = (~y_true[y_pred]).sum()

        return {'val_loss': loss, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

    def validation_end(self, outputs):
        tp = sum([o['tp'] for o in outputs])
        fp = sum([o['fp'] for o in outputs])
        tn = sum([o['tn'] for o in outputs])
        fn = sum([o['fn'] for o in outputs])

        loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) else 0.
        rec = tp / (tp + fn) if (tp + fn) else 0.
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0.

        return {
            'val_loss': loss,
            'val_acc': acc,
            'val_prec': prec,
            'val_rec': rec,
            'val_f1': f1
        }

    @pl.data_loader
    def train_dataloader(self):
        dataset = loader.ImageLoader.get_train_loader(self.config,
                                                      self.normalize)
        sampler_cls = (torch.utils.data.DistributedSampler
                       if self.distributed_sampler else
                       torch.utils.data.SequentialSampler)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['data']['batch_size'],
            sampler=sampler_cls(dataset),
            num_workers=5,
            pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        dataset = loader.ImageLoader.get_val_loader(self.config, self.normalize)

        sampler_cls = (torch.utils.data.DistributedSampler
                       if self.distributed_sampler else
                       torch.utils.data.SequentialSampler)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['data']['batch_size'],
            sampler=sampler_cls(dataset),
            num_workers=5,
            pin_memory=True)

    def configure_optimizers(self):
        optim_conf = self.config['optim']
        optim_cls = optim.__dict__[optim_conf['type']]
        return optim_cls(self.parameters(), **optim_conf['kwargs'])

    def configure_loss(self):
        loss_conf = self.config['loss']
        loss_cls = nn.modules.loss.__dict__[loss_conf['type']]
        return loss_cls(**loss_conf['kwargs'])


def run(config):
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)

    trainer_conf = config['trainer']
    experiment = Experiment(config)
    trainer = pl.Trainer(**trainer_conf)
    trainer.fit(experiment)
