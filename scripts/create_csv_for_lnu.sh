#!/bin/bash

# List of models
models=("ADM" "DDPM" "Diff-ProjectedGAN" "Diff-StyleGAN2" "IDDPM" "LDM" "PNDM" "ProGAN" "ProjectedGAN" "StyleGAN")

# Base command
cmd="scripts/create_csv.py --original-data-path \"/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/diffusion_model_deepfakes_lsun_bedroom/test/Real\" --edited-data-path \"/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/diffusion_model_deepfakes_lsun_bedroom/test/\" --base-path \"/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/csv/\" --read-key \"/*.png\" --data-type"

# Iterate over each model and execute the command
for model in "${models[@]}"; do
    echo "Executing for model: $model"
    eval "$cmd $model"
done
