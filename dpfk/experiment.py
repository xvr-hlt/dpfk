import glob
import os
import types
from os import path

import pytorch_lightning as pl
import torch
import yaml
from torch import distributed, nn, optim
from torch.nn import functional as F
from torchvision import transforms

import dpfk.nn.model
import wandb
from dpfk.data import loader, util


class Experiment(pl.LightningModule):

    ncpus = 15

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model, self.normalize = dpfk.nn.model.get_model_normalize_from_config(
            config)
        self.loss = self.configure_loss()

        self.distributed_backend = config['trainer'].get('distributed_backend')
        self.distributed_sampler = (self.distributed_backend == 'ddp')

        batch_size = config['data']['batch_size']

        if self.distributed_backend == "dp":
            gpus = config['trainer']['gpus']
            batch_size *= gpus

        self._batch_size = None
        self._wandb = None
        self._rank = None

    @property
    def batch_size(self):
        if self._batch_size is None:
            batch_size = self.config['data']['batch_size']
            if self.trainer.use_dp:
                batch_size *= self.config['trainer']['gpus']
            self._batch_size = batch_size
        return self._batch_size

    def forward(self, x):
        return self.model.forward(x)

    @property
    def wandb(self):
        if self._wandb is None:
            self._wandb = wandb.init(project='dpfk', config=self.config)
        return self._wandb

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        return {'loss': loss}

    @property
    def rank(self):
        if self._rank is None:
            try:
                self._rank = distributed.get_rank()
            except AssertionError:
                self._rank = 0
        return self._rank

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        y_pred = y_hat > 0
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
        tp = torch.stack([o['tp'] for o in outputs]).sum().float()
        fp = torch.stack([o['fp'] for o in outputs]).sum().float()
        tn = torch.stack([o['tn'] for o in outputs]).sum().float()
        fn = torch.stack([o['fn'] for o in outputs]).sum().float()

        n = (tp + tn + fp + fn)

        loss = torch.stack([o['val_loss'] for o in outputs]).mean()
        acc = (tp + tn) / n
        prec = tp / (tp + fp) if (tp + fp) else 0.
        rec = tp / (tp + fn) if (tp + fn) else 0.
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0.

        metrics = {
            'val_loss': loss,
            'val_acc': acc,
            'val_prec': prec,
            'val_rec': rec,
            'val_f1': f1,
            'val_n': n
        }

        if self.rank == 0:
            self.wandb.log(metrics)

        return metrics

    @pl.data_loader
    def train_dataloader(self):
        dataset = loader.ImageLoader.get_train_loader(self.config,
                                                      self.normalize)
        if self.trainer.use_ddp:
            sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           sampler=sampler,
                                           num_workers=self.ncpus,
                                           pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        dataset = loader.ImageLoader.get_val_loader(self.config, self.normalize)

        if self.trainer.use_ddp:
            sampler = torch.utils.data.DistributedSampler(dataset,
                                                          shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           sampler=sampler,
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


def run(config):
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)
    trainer_conf = config['trainer']
    experiment = Experiment(config)
    trainer = pl.Trainer(**trainer_conf)
    trainer.fit(experiment)
