import argparse
from typing import Any, Dict, List


import yaml
from loader.build import ImageLoadingModule, TensorLoadingModule

import evaluate
import torch

metric = evaluate.load("accuracy")
from pytorch_lightning import Trainer
from model.build import build_model
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Any, Dict
from core.storage import ensure_dir, create_path
from core.fft import get_mask_fn
from loader.transforms import BandFilter, FDAFilter


def run_train(args):
    if args.fast_dev_run:
        args.batch_size = 5
    torch.set_float32_matmul_precision("medium")
    mask_func = get_mask_fn(args.filter_type)
    mask = mask_func(width=args.image_size, height=args.image_size, n=1, D0=30)
    dataModule = ImageLoadingModule(
        type=args.data_type,
        set=args.data_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        test_data_path=args.test_data_path,
        # filter=BandFilter(mask),
    )
    dataModule.setup()
    args.loss_weight = torch.Tensor([dataModule.train_data.ratio])

    model = build_model(
        encoder_type=args.encoder_type,
        checkpoint=args.encoder_checkpoint,
        hidden_size=args.hidden_size,
        head_type=args.head_type,
        num_labels=args.num_labels,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
    )
    name = create_model_name(args)
    logger = TensorBoardLogger("tensorboard_logs", name=name)
    model_name = name + "{epoch}-{val_loss:.2f}"
    joined_path = create_path(
        "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/model/checkpoints",
        args.experiment_name,
    )
    ensure_dir(joined_path)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=joined_path,
        filename=model_name,
        save_top_k=1,
        mode="min",
    )
    trainer = Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=20,
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=args.acc_grad_batch,
        # accelerator="gpu",
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        logger=logger,
        precision="16",
        limit_train_batches=0.75,
        limit_val_batches=0.1,
        val_check_interval=0.25,
        # profiler="simple",#AdvancedProfiler(dirpath="/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/results/", filename="profiler")
    )

    trainer.fit(model, datamodule=dataModule)


def create_model_name(args):
    return f"{args.encoder_type}_{args.data_type}_{args.hidden_size}_{args.head_type}"


def main(args):
    """This function starts the main Experiment Pipeline."""
    run_train(args)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure the evaluation pipeline")

    parser.add_argument("--experiment_name", "-en", default="supContrastive", type=str)
    parser.add_argument("--from_tensor", default=False, action="store_true")
    parser.add_argument("--model_name", "-mn", default="Dino", type=str)
    parser.add_argument(
        "--model_type",
        default="image",
        type=str,
        choices=["image_grounded", "image", "image_sam"],
    )
    parser.add_argument("--batch_size", default=128, type=int)

    parser.add_argument(
        "--data_type", default="coco", type=str, choices=["coco", "faces"]
    )
    parser.add_argument("--data_set", default="coco", type=str)
    parser.add_argument(
        "--train_data_path",
        default="/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/",
        type=str,
    )
    parser.add_argument(
        "--test_data_path",
        default="/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/",
        type=str,
    )
    parser.add_argument(
        "--val_data_path",
        default="/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/",
        type=str,
    )

    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=0.005, type=float)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--logging_steps", default=5, type=int)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--num_labels", "-nm", default=1024, type=int)
    parser.add_argument("--acc_grad_batch", default=1, type=int)
    parser.add_argument("--fast_dev_run", default=False, action="store_true")
    parser.add_argument(
        "--loss_type", "-lt", default="BCE", type=str, choices=["SubConLoss", "BCE"]
    )
    parser.add_argument("--head_type", "-ht", default="linear", type=str)
    parser.add_argument(
        "--image_mean", default=[0.485, 0.456, 0.406], nargs="+", type=float
    )
    parser.add_argument(
        "--image_std", default=[0.229, 0.224, 0.225], nargs="+", type=float
    )
    parser.add_argument("--image_size", default=224, type=int)

    parser.add_argument("--test", "-t", default=False, action="store_true")
    parser.add_argument("--hidden_size", "-hs", default=1024, type=int)

    parser.add_argument("--has_transform", default=False, action="store_true")
    parser.add_argument(
        "--filter_type",
        "-ft",
        type=str,
        required=False,
        choices=["butterworth_lowpass", "butterworth_highpass", "butterworth_bandpass"],
        help="Type of filter to use",
    )
    parser.add_argument("--band_width", default=10, type=int)
    parser.add_argument("--filter_radius", default=5, type=int)

    parser.add_argument("--encoder_type", "-et", default="DinoV2", type=str)
    parser.add_argument("--encoder_checkpoint", "-ec", default=None, type=str)

    parser.add_argument("--model_checkpoint", "-mc", default=None, type=str)
    parser.add_argument(
        "--save_path", "-sp", type=str, required=True, help="Path to save results"
    )
    parser.add_argument(
        "--for_all", action="store_true", help="Flag to process all data types"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    data_types = [
        "ProGAN",
        # "all",
        # "gan",
        # "diff",
        # "ADM",
        # "coco",
        # "DDPM",
        # "Diff-ProjectedGAN",
        # "Diff-StyleGAN2",
        # "IDDPM",
        # "LDM",
        # "PNDM",
        # "ProjectedGAN",
        # "StyleGAN",
    ]

    if args.for_all:
        for data_type in data_types:
            args.data_type = data_type
            args.data_set = "csv"
            main(args)
    else:
        main(args)
