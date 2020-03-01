import fire
import pytorch_lightning
import yaml

from . import experiment

fire.Fire(experiment.run)
