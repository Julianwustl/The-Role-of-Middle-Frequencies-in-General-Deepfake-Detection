
from typing import Any, Dict


import yaml
from loader.build import ImageLoadingModule

import evaluate
import torch
metric = evaluate.load("accuracy")
from pytorch_lightning import Trainer
from model.build import build_model
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from train import run_train,run_on_all_models
from MasterThesis.core.fft import create_masks


models = {
    "DinoV2": ["facebook/dinov2-large","facebook/dinov2-giant"],
    "OpenClip":["laion2b_s34b_b88k"], 
    
}
def create_configs(config: Dict[str, Any]):
    configs = []
    for encoder, paths in models.items():
        for path in paths:
            config["image"]["encoder_type"] = encoder
            config["image"]["checkpoint"] = path
            configs.append(config)
    return configs


def main(config: Dict[str, Any]):
    """This function starts the main Experiment Pipeline. """

    # Runs the config over all models.
    configs = create_configs(config)
    for config in configs:
        run_train(config)
    masks = create_masks(config["image_size"], config["band_width"])
    for mask in masks:
        config["fft_mask"] = mask
        # Run evaluation
    
    


