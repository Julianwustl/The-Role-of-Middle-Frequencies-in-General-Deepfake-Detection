import json
from loader.build import ImageLoadingModule
import evaluate
import torch
metric = evaluate.load("accuracy")
from pytorch_lightning import Trainer
from model.build import build_model

from lightning.pytorch.loggers import CSVLogger
from core.fft import get_mask_fn
from loader.transforms import BandFilter
from callbacks import EmbeddingCollectorCallback
from core.storage import ensure_dir, create_path
import argparse
import os


def run_evaluation(args, filter=None, radius=None):

    torch.set_float32_matmul_precision('medium')
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

    )
    dataModule.setup()
   
    trainer = Trainer(
        max_epochs=args.num_epochs,
        fast_dev_run=args.fast_dev_run,
        enable_progress_bar=True,
        accelerator="gpu",
        # callbacks=[
        #     EmbeddingCollectorCallback(
        #         path=create_path(args.save_path,"embedding",args.experiment_name),
        #         file_name=f"{args.encoder_type}_{args.data_type}_{args.filter_type}_{radius}")
        #         ]
    )
    
    loaded_model = build_model(
        encoder_type=args.encoder_type,
        checkpoint=args.encoder_checkpoint,
        hidden_size=args.hidden_size,
        head_type=args.head_type,
        num_labels=args.num_labels,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        extract=True,
        #filter=filter,


    )  # Load your model here
    
    return  trainer.predict(model=loaded_model, datamodule=dataModule) 


def run_with_filter(args,base_path:str):
    """This function starts the main Experiment Pipeline. """
    test_results = []
    mask_func = get_mask_fn(args.filter_type)
    for radius in range(0,90,args.filter_radius):
        if radius == 0:
            radius = 1
        mask = mask_func(width=args.image_size, height=args.image_size,n=1,D0=radius)
        # We have to save the data manually, so we can run the proper analytics. 
        
        result = run_evaluation(args, filter=BandFilter(mask), radius = radius)
        # Get the feature spacec
        # feature_space = result.get("test_embedding")
        #torch.save(all_embeddings, os.path.join(new_path,f"{self.file_name}_encoder.pt"))
        save_embeddings(result, base_path, f"{args.encoder_type}_{args.data_type}_{args.filter_type}_{radius}")
        test_results.append({radius: result})
        # Run evaluation
    
    return test_results

def run_without_filter(args,base_path:str):
    
    result = run_evaluation(args,radius="no")
    
    
    save_embeddings(result, base_path, f"{args.encoder_type}_{args.data_type}_{args.filter_type}")

def save_embeddings(results, base_path:str, file_name:str):
    embeddings = {}
    for i, embedding_dict in enumerate(results):

        for i, label in enumerate(embedding_dict["label"]):
            label = str(label.to("cpu").item())
            if label not in embeddings:
                embeddings[label] = [embedding_dict["embedding"][i]]
                continue
            embeddings[label].append(embedding_dict["embedding"][i])

    for label, tensors in embeddings.items():
        all_embeddings = torch.stack(tensors, dim=0)
        new_path = create_path(base_path,label)
        ensure_dir(new_path)
        torch.save(all_embeddings, os.path.join(new_path,f"{file_name}_linear.pt"))

def main(args):
    base_path = create_path(args.save_path,"embedding",args.experiment_name,args.extraction_type,args.data_type)
    if args.filter_type is None:
        run_without_filter(args,base_path)
    else:

        run_with_filter(args,base_path)
    

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure the evaluation pipeline")

    parser.add_argument("--experiment_name","-en", default="supContrastive", type=str)
    parser.add_argument("--model_name", "-mn", default="Dino", type=str)
    parser.add_argument("--model_type", default="image", type=str, choices=["image_grounded", "image", "image_sam"])
    parser.add_argument("--batch_size", default=128, type=int)
    
    parser.add_argument("--data_type", default="coco", type=str, choices=["coco", "faces"])
    parser.add_argument("--data_set", default="coco", type=str)
    parser.add_argument("--train_data_path", default="/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data", type=str)
    parser.add_argument("--test_data_path", default="/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data", type=str)
    parser.add_argument("--val_data_path", default="/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data", type=str)
    parser.add_argument("--extraction_type", default="train", type=str, choices=['train', 'val', 'test'],)
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=0.0005, type=float)
    parser.add_argument("--warmup_ratio", default=0, type=float)
    parser.add_argument("--logging_steps", default=5, type=int)
    parser.add_argument("--max_steps", default=None, type=int)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--num_labels", "-nm", default=1024, type=int)
    parser.add_argument("--acc_grad_batch", default=1, type=int)
    parser.add_argument("--fast_dev_run", default=False, action="store_true")
    parser.add_argument("--loss_type", default="SubConLoss", type=str, choices=["SubConLoss", "BCE"])
    parser.add_argument("--head_type","-ht", default="linear", type=str)
    parser.add_argument("--image_mean", default=[0.485, 0.456, 0.406], nargs="+", type=float)
    parser.add_argument("--image_std", default=[0.229, 0.224, 0.225], nargs="+", type=float)
    parser.add_argument("--image_size", default=224, type=int)
    
    parser.add_argument("--test","-t", default=True, action="store_true")
    parser.add_argument("--hidden_size", "-hs", default=1024, type=int)
    
    parser.add_argument("--has_transform", default=False, action="store_true")
    parser.add_argument('--filter_type',"-ft",type=str,default=None, required=False, choices=['butterworth_lowpass', 'butterworth_highpass', 'FDA'], help='Type of filter to use')
    parser.add_argument("--band_width", default=10, type=int)
    parser.add_argument("--filter_radius", default=5, type=int)
    
    parser.add_argument("--encoder_type", "-et",default="DinoV2", type=str)
    parser.add_argument("--encoder_checkpoint", "-ec" ,default=None, type=str)
    
    parser.add_argument("--model_checkpoint","-mc",default=None, type=str)
    parser.add_argument('--save_path',"-sp",type=str, required=True, help='Path to save results')
    parser.add_argument("--nrows", default=1000, type=int)
    parser.add_argument('--for_all', action='store_true', help='Flag to process all data types')

    return parser.parse_args()

    # List of data types 
    

if __name__ == "__main__":
    args = parse_arguments()
    data_types = [ "Diff-ProjectedGAN", "Diff-StyleGAN2", "IDDPM", "LDM", "PNDM", "ProGAN", "ProjectedGAN", "StyleGAN","ADM", "DDPM","coco"]

    if args.for_all:
        for data_type in data_types:
            args.data_type = data_type
            args.data_set = "csv"
            main(args)
    else:
        main(args)
