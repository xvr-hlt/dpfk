import glob
import os
import types
from os import path

import neptune
import pytorch_lightning as pl
import torch
import yaml
from kornia import augmentation
from pytorch_lightning.logging import wandb
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
        self.augment = self.configure_augment()

        self.distributed_backend = config['trainer'].get('distributed_backend')
        self.distributed_sampler = (self.distributed_backend == 'ddp')

        batch_size = config['data']['batch_size']
        ncpus = 20
        if self.distributed_backend == "dp":
            gpus = config['trainer']['gpus']
            batch_size *= gpus
            # ncpus *= gpus
        self.batch_size = batch_size
        self.ncpus = ncpus

    def forward(self, x):
        if self.training:
            x = self.augment(x)
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        return {'loss': loss}

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

        state = {'val_loss': loss, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

        if self.trainer.use_dp or self.trainer.use_ddp2:
            state = {k: v.unsqueeze(0) for k, v in state.items()}

        return state

    def validation_end(self, outputs):
        tp = torch.stack([o['tp'] for o in outputs]).sum()
        fp = torch.stack([o['fp'] for o in outputs]).sum()
        tn = torch.stack([o['tn'] for o in outputs]).sum()
        fn = torch.stack([o['fn'] for o in outputs]).sum()

        loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) else 0.
        rec = tp / (tp + fn) if (tp + fn) else 0.
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0.

        metrics = {
            'val_loss': loss,
            'val_acc': acc,
            'val_prec': prec,
            'val_rec': rec,
            'val_f1': f1
        }
        return metrics

    @pl.data_loader
    def train_dataloader(self):
        dataset = loader.ImageLoader.get_train_loader(self.config,
                                                      self.normalize)
        sampler_cls = (torch.utils.data.DistributedSampler
                       if self.distributed_sampler else
                       torch.utils.data.SequentialSampler)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           sampler=sampler_cls(dataset),
                                           num_workers=self.ncpus,
                                           pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        dataset = loader.ImageLoader.get_val_loader(self.config, self.normalize)

        sampler_cls = (torch.utils.data.DistributedSampler
                       if self.distributed_sampler else
                       torch.utils.data.SequentialSampler)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           sampler=sampler_cls(dataset),
                                           num_workers=self.ncpus,
                                           pin_memory=True)

    def configure_optimizers(self):
        optim_conf = self.config['optim']
        optim_cls = optim.__dict__[optim_conf['type']]
        return optim_cls(self.parameters(), **optim_conf['kwargs'])

    def configure_loss(self):
        loss_conf = self.config['loss']
        loss_cls = nn.modules.loss.__dict__[loss_conf['type']]
        return loss_cls(**loss_conf['kwargs'])

    def configure_augment(self):
        prob = self.config['aug']['prob']
        aug_level = self.config['aug']['level']
        augments = []
        augments.append(augmentation.RandomHorizontalFlip(prob))
        #degrees = [15., 30., 45.][aug_level]
        #augments.append(augmentation.RandomRotation(degrees))
        #distortion_scale = [.1, .25, .5][aug_level]
        #augments.append(augmentation.RandomPerspective(distortion_scale,
        #                                               p=prob))
        return nn.Sequential(*augments)


def run(config):
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)
    trainer_conf = config['trainer']

    logger = wandb.WandbLogger(project="dpfk")
    logger.experiment.config = config
    experiment = Experiment(config)
    trainer = pl.Trainer(**trainer_conf)
    trainer.fit(experiment)
