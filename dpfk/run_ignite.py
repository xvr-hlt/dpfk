import fire
import ignite
import numpy as np
import torch
import yaml
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import Accuracy, Loss
from torch import distributed, nn, optim
from tqdm.auto import tqdm

import dpfk
from dpfk import experiment
from dpfk.data import loader

log_interval = 1


class RandomSubset(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, dataset, n):
        self.N = len(dataset)
        self.n = n

    def __iter__(self):
        return iter(np.random.randint(0, self.N, self.n))

    def __len__(self):
        return self.n


def ignite_train(config):
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)

    model, normalize = dpfk.nn.model.get_model_normalize_from_config(config)
    model = nn.DataParallel(model)

    tdataset = loader.ImageLoader.get_train_loader(config, normalize)

    train_loader = torch.utils.data.DataLoader(
        tdataset,
        batch_size=config['data']['batch_size'] * 2,
        num_workers=20,
        sampler=RandomSubset(tdataset, 20000),
        pin_memory=True)

    vdataset = loader.ImageLoader.get_val_loader(config, normalize)
    val_loader = torch.utils.data.DataLoader(
        vdataset,
        batch_size=config['data']['batch_size'],
        num_workers=20,
        pin_memory=True)

    optim_conf = config['optim']
    optim_cls = optim.__dict__[optim_conf['type']]
    optimizer = optim_cls(model.parameters(), **optim_conf['kwargs'])

    loss_conf = config['loss']
    loss_cls = nn.modules.loss.__dict__[loss_conf['type']]
    loss_fn = loss_cls(**loss_conf['kwargs'])

    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        loss_fn,
                                        device="cuda")

    def output_transform(output):
        y_pred, y = output
        y_pred = (y_pred > 0).int()
        return y_pred, y

    evaluator = create_supervised_evaluator(
        model,
        metrics={
            "accuracy": Accuracy(output_transform=output_transform),
            "nll": Loss(loss_fn)
        },
        device="cuda")

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0,
                leave=False,
                total=len(train_loader),
                desc=desc.format(0))

    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)
        pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_nll))

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=config['trainer']['max_epochs'])
    pbar.close()


if __name__ == "__main__":
    fire.Fire(ignite_train)
