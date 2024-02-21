import os
import json
import time
from loader.build import ImageLoadingModule
import evaluate
import torch

metric = evaluate.load("accuracy")
from pytorch_lightning import Trainer
from model.build import build_model
from lightning.pytorch.loggers import CSVLogger
from core.fft import get_mask_fn
from loader.transforms import BandFilter, FDAFilter
from core.storage import ensure_dir, create_path
import argparse
import pandas as pd


def run_evaluation(args, filter=None, radius=None):
    torch.set_float32_matmul_precision("medium")
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
        filter=filter,
        nrows=args.nrows,
    )
    dataModule.setup()

    trainer = Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=args.acc_grad_batch,
        enable_progress_bar=True,
        accelerator="gpu",
        precision="16",
        logger=CSVLogger(save_dir="logs/"),
    )
    start_time = time.time()
    loaded_model = build_model(
        encoder_type=args.encoder_type,
        checkpoint=args.encoder_checkpoint,
        hidden_size=args.hidden_size,
        head_type=args.head_type,
        num_labels=args.num_labels,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        # filter=filter,
    )  # Load your model here
    end_time = time.time()  # End time after the function is called

    elapsed_time = end_time - start_time
    print(f"The function took {elapsed_time} seconds to execute.")
    return trainer.test(
        model=loaded_model, datamodule=dataModule, ckpt_path=args.model_checkpoint
    )


def run_with_filter(args):
    """This function starts the main Experiment Pipeline."""
    test_results = []
    torch.set_float32_matmul_precision("medium")

    mask_func = get_mask_fn(args.filter_type)
    trainer = Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=args.acc_grad_batch,
        enable_progress_bar=True,
        accelerator="gpu",
        precision="16",
        logger=CSVLogger(save_dir="logs/"),
    )
    loaded_model = build_model(
        encoder_type=args.encoder_type,
        checkpoint=args.encoder_checkpoint,
        hidden_size=args.hidden_size,
        head_type=args.head_type,
        num_labels=args.num_labels,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        # filter=filter,
    )
    loaded_model.load_state_dict(torch.load(args.model_checkpoint)["state_dict"])
    for radius in range(0, 190, args.filter_radius):
        if radius == 0:
            radius = 1

        mask = mask_func(width=args.image_size, height=args.image_size, n=1, D0=radius)
        # We have to save the data manually, so we can run the proper analytics.
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
            filter=BandFilter(mask),
            nrows=args.nrows,
        )
        dataModule.setup()
        result = trainer.test(
            model=loaded_model,
            datamodule=dataModule,
            # ckpt_path=args.model_checkpoint,
        )
        # Get the feature spacec
        # feature_space = result.get("test_embedding")

        test_results.append({radius: result})

    return test_results


def transform_fda_target(tensor):
    tensor = torch.from_numpy(tensor)
    tensor = tensor.permute(2, 0, 1)
    return tensor


def read_csv(path):
    return pd.read_csv(path, delimiter=" ", names=["image", "label"]).to_dict(
        orient="records"
    )


def run_with_fda(args):
    """This function starts the main Experiment Pipeline."""
    refernce_images = read_csv(
        f"{args.test_data_path}/{args.data_set}/test_data_{args.data_type}.csv"
    )
    beta = 0.01
    result_image = run_evaluation(
        args, filter=FDAFilter(refernce_images, beta=beta), radius=beta
    )

    # feature_space = result.get("test_embedding")
    return {"FDA": result_image}


def run_without_filter(config):
    result = run_evaluation(config, radius="no")
    return {"no": result}


def main(args):
    path = create_path(args.save_path)
    results_no_filter = run_without_filter(args)

    results_filter = run_with_filter(args)
    results_filter.append(results_no_filter)
    results_path = create_path(path, "classification", args.experiment_name)
    ensure_dir(results_path)
    with open(
        f"{results_path}/{args.encoder_type}_{args.filter_type}_{args.data_type}.json",
        "w",
    ) as f:
        json.dump(results_filter, f)


def run_all(args):
    """This Function runs all models against all datasets."""
    data_types = [
        "all",
        "gan",
        "diff",
        "ADM",
        "coco",
        "DDPM",
        "Diff-ProjectedGAN",
        "Diff-StyleGAN2",
        "IDDPM",
        "LDM",
        "PNDM",
        "ProGAN",
        "ProjectedGAN",
        "StyleGAN",
    ]
    path = "/".join(args.model_checkpoint.split("/")[:-1])
    model_name = args.model_name
    files = []
    for subdirectory in os.listdir(path):
        file = os.path.join(path, subdirectory)
        if os.path.isfile(file) and model_name in file:
            files.append(file)
    for file in files:
        data_trained_on = file.split("/")[-1].split("_")[1]
        if data_trained_on in ["gan", "diff"]:
            continue
        for data_type in data_types:
            print(f"Running {data_type} on {data_trained_on}")
            args.data_type = data_type
            args.data_set = "csv"
            args.model_checkpoint = file
            path = create_path(args.save_path)
            if args.filter_type == "FDA":
                results = run_with_fda(args)
            else:
                results_no_filter = run_without_filter(args)
                results = run_with_filter(args)
                results.append(results_no_filter)

            results_path = create_path(
                path, "classification", args.experiment_name, data_trained_on
            )
            ensure_dir(results_path)
            with open(
                f"{results_path}/{args.encoder_type}_{args.filter_type}_{args.data_type}.json",
                "w",
            ) as f:
                json.dump(results, f)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure the evaluation pipeline")

    parser.add_argument("--experiment_name", "-en", default="supContrastive", type=str)
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
        default="/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data",
        type=str,
    )
    parser.add_argument(
        "--test_data_path",
        default="/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data",
        type=str,
    )
    parser.add_argument(
        "--val_data_path",
        default="/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data",
        type=str,
    )

    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=0.0005, type=float)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--logging_steps", default=5, type=int)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--num_labels", "-nm", default=1024, type=int)
    parser.add_argument("--acc_grad_batch", default=1, type=int)
    parser.add_argument("--fast_dev_run", default=False, action="store_true")
    parser.add_argument(
        "--loss_type", default="SubConLoss", type=str, choices=["SubConLoss", "BCE"]
    )
    parser.add_argument("--head_type", "-ht", default="linear", type=str)
    parser.add_argument(
        "--image_mean", default=[0.485, 0.456, 0.406], nargs="+", type=float
    )
    parser.add_argument(
        "--image_std", default=[0.229, 0.224, 0.225], nargs="+", type=float
    )
    parser.add_argument("--image_size", default=224, type=int)

    parser.add_argument("--test", "-t", default=True, action="store_true")
    parser.add_argument("--hidden_size", "-hs", default=1024, type=int)

    parser.add_argument("--has_transform", default=False, action="store_true")
    parser.add_argument(
        "--filter_type",
        "-ft",
        type=str,
        required=True,
        choices=[
            "butterworth_lowpass",
            "butterworth_highpass",
            "butterworth_bandpass",
            "FDA",
        ],
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
    parser.add_argument("--nrows", default=5000, type=int)
    parser.add_argument(
        "--for_all", action="store_true", help="Flag to process all data types"
    )

    return parser.parse_args()

    # List of data types


if __name__ == "__main__":
    args = parse_arguments()
    if args.for_all:
        run_all(args)
    else:
        main(args)
