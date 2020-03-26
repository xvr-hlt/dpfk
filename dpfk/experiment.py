import datetime
import glob
import os
import random
import types
from os import path

import pytorch_lightning as pl
import torch
import wandb
import yaml
from pytorch_lightning import callbacks
from torch import distributed, nn, optim
from torch.nn import functional as F
from torchvision import transforms

import dpfk.nn.model
from dpfk.data import loader, util


class Experiment(pl.LightningModule):

    NCPUS = 15

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_scales = config['image_scales']

        self.model, self.normalize = dpfk.nn.model.get_model_normalize_from_config(
            config)

        self.loss = self.configure_loss()

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
        with torch.no_grad():
            scale = random.choice(self.image_scales)
            x = F.interpolate(x, scale_factor=scale)
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

        tp = (y_true * y_pred).sum()
        fp = (~y_true * y_pred).sum()
        fn = (y_true * ~y_pred).sum()
        tn = (~y_true * ~y_pred).sum()

        val_loss_pos = F.binary_cross_entropy_with_logits(y_hat[y_true],
                                                          y[y_true],
                                                          reduction='sum')
        val_loss_neg = F.binary_cross_entropy_with_logits(y_hat[~y_true],
                                                          y[~y_true],
                                                          reduction='sum')

        state = {
            'val_loss': loss,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'val_loss_pos': val_loss_pos,
            'val_loss_neg': val_loss_neg,
        }

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

        loss_neg = torch.stack([o['val_loss_neg'] for o in outputs]).sum()
        loss_neg /= (tn + fp)

        loss_pos = torch.stack([o['val_loss_pos'] for o in outputs]).sum()
        loss_pos /= (tp + fn)

        loss_weighted = (loss_neg + loss_pos) / 2

        tpr = tp / (tp + fn) if (tp + fn) else 0.
        tnr = tn / (tn + fp) if (tn + fp) else 0.

        metrics = {
            'val_loss': loss,
            'val_loss_weighted': loss_weighted,
            'val_loss_pos': loss_pos,
            'val_loss_neg': loss_neg,
            'val_tpr': tpr,
            'val_tnr': tnr,
            'val_balanced_acc': (tpr + tnr) / 2,
            'val_n': n,
            'val_tp': tp,
            'val_fp': fp,
            'val_tn': tn,
            'val_fn': fn
        }

        if self.rank == 0 and n > 1000:
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
                                           num_workers=self.NCPUS,
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
                                           num_workers=self.NCPUS,
                                           pin_memory=True)

    def configure_optimizers(self):
        optim_conf = self.config['optim']
        optim_cls = optim.__dict__[optim_conf['type']]
        optimizer = optim_cls(self.parameters(), **optim_conf['kwargs'])

        optim_scheduler_conf = self.config['optim_scheduler']
        optim_scheduler_cls = optim.lr_scheduler.__dict__[
            optim_scheduler_conf['type']]
        optim_scheduler = optim_scheduler_cls(optimizer,
                                              **optim_scheduler_conf['kwargs'])

        return [optimizer], [optim_scheduler]

    def configure_loss(self):
        loss_conf = self.config['loss']
        loss_cls = nn.modules.loss.__dict__[loss_conf['type']]
        loss_kwargs = loss_conf['kwargs']
        if 'pos_weight' in loss_kwargs:
            loss_kwargs['pos_weight'] = torch.Tensor(loss_kwargs['pos_weight'])
        return loss_cls(**loss_kwargs)


def run(config):
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)

    now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = path.join("wandb", now)
    run_dir = path.abspath(run_dir)
    os.environ['WANDB_RUN_DIR'] = run_dir

    checkpoint_callback = callbacks.ModelCheckpoint(
        run_dir, monitor=config['early_stopping']['monitor'])
    early_stopping_callback = callbacks.EarlyStopping(
        **config['early_stopping'])

    experiment = Experiment(config)
    trainer = pl.Trainer(logger=False,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=early_stopping_callback,
                         **config['trainer'])
    trainer.fit(experiment)
