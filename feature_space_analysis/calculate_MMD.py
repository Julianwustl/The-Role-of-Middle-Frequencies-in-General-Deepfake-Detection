# Proposed Refactored Code

import argparse
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE

# Constants
DEFAULT_BASE_INPUT_FOLDER = "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/results/embeddings/"
FEATURE_DIR = "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/results/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Kernel:
    """Base class for kernels

    Unless otherwise noted, all kernels implementing lengthscale detection
    use the median of pairwise distances as the lengthscale."""

    pass


class GaussianKernel(Kernel):
    r"""Unnormalized gaussian kernel

    .. math::
        k(|x-y|) = \exp(-|x-y|^2/(2\ell^2))

    where :math:`\ell` is the `lengthscale` (autodetected or given)."""

    def __init__(self, lengthscale=None):
        super().__init__()
        self.lengthscale = lengthscale

    def __call__(self, dists):
        # note that lengthscale should be squared in the RBF to match the Gretton et al heuristic
        if self.lengthscale is not None:
            lengthscale = self.lengthscale
        else:
            lengthscale = dists.median()
        return torch.exp((-0.5 / lengthscale ** 2) * dists ** 2)


class ExpKernel(Kernel):
    r"""Unnormalized exponential kernel

    .. math::
        k(|x-y|) = \exp(-|x-y|/\ell)

    where :math:`\ell` is the `lengthscale` (autodetected or given)."""

    def __init__(self, lengthscale=None):
        super().__init__()
        self.lengthscale = lengthscale

    def __call__(self, dists):
        if self.lengthscale is not None:
            lengthscale = self.lengthscale
        else:
            lengthscale = dists.median()
        return torch.exp((-1 / lengthscale) * dists)


class RationalQuadraticKernel(Kernel):
    r"""Unnormalized rational quadratic kernel

    .. math::
        k(|x-y|) = (1+|x-y|^2/(2 \alpha \ell^2))^{-\alpha}

    where :math:`\ell` is the `lengthscale` (autodetected or given)."""

    def __init__(self, lengthscale=None, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.lengthscale = lengthscale

    def __call__(self, dists):
        if self.lengthscale is not None:
            lengthscale = self.lengthscale
        else:
            lengthscale = dists.median()
        return torch.pow(
            1 + (1 / (2 * self.alpha * lengthscale ** 2)) * dists ** 2, -self.alpha
        )

def MMD(x, y, kernel=GaussianKernel(),n_perm=1000):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    n, d = x.shape
    m, d2 = y.shape
    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    k = kernel(dists)
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = (
        k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    )
    if n_perm is None:
        return mmd
    # mmd_0s = []
    # count = 0
    # for i in range(n_perm):
    #     # this isn't efficient, it would be lovely to generate a cuda kernel or C++ for loop and do the
    #     # permutation on the fly...
    #     pi = torch.randperm(n + m, device=x.device)
    #     k = k[pi][:, pi]
    #     k_x = k[:n, :n]
    #     k_y = k[n:, n:]
    #     k_xy = k[:n, n:]
    #     # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    #     mmd_0 = (
    #         k_x.sum() / (n * (n - 1))
    #         + k_y.sum() / (m * (m - 1))
    #         - 2 * k_xy.sum() / (n * m)
    #     )
    #     mmd_0s.append(mmd_0)
    #     count = count + (mmd_0 > mmd)
    # # pyplot.hist(torch.stack(mmd_0s, dim=0).tolist(), bins=50)
    # # true_divide: torch 1.6 compat replace with "/" after October 2021
    # mmds = torch.tensor(mmd_0s, device=x.device)
    # print(torch.quantile(mmds, 0.95))
    # print((mmd < mmds).float().mean())
    # p_val = torch.true_divide(count, n_perm)
    # print(p_val)
    return mmd
    return torch.mean(XX + YY - 2. * XY)


def get_files_from_sub_directory(path):
    """Retrieve all files from a specified directory. If there is a sub directory, create a new key in the dict"""
    files = {}
    for subdirectory in os.listdir(path):
        full_subdirectory_path = os.path.join(path, subdirectory)
        if os.path.isdir(full_subdirectory_path):  # Ensure it's a directory
            if subdirectory not in files:
                files[subdirectory] = {}
            for subsubdirectory in os.listdir(full_subdirectory_path):
                full_subsubdirectory_path = os.path.join(full_subdirectory_path, subsubdirectory)
                if os.path.isdir(full_subsubdirectory_path):
                    files[subdirectory][subsubdirectory] = [os.path.join(full_subsubdirectory_path, file) for file in os.listdir(full_subsubdirectory_path) if os.path.isfile(os.path.join(full_subsubdirectory_path, file))]
    return files


def get_files_from_directory(path):
    """Retrieve all files from a specified directory. If there is a sub directory, create a new key in the dict"""
    files = {}
    for subdirectory in os.listdir(path):
        full_subdirectory_path = os.path.join(path, subdirectory)
        if os.path.isdir(full_subdirectory_path):  # Ensure it's a directory
            files[subdirectory] = [os.path.join(full_subdirectory_path, file) for file in os.listdir(full_subdirectory_path) if os.path.isfile(os.path.join(full_subdirectory_path, file))]
    return files


def split_and_filter_files(files):
    """Split files into real and model-generated embeddings."""
    real_embeddings = [file for file in files if "real" in file]
    model_embeddings = [file for file in files if "real" not in file]
    return real_embeddings, model_embeddings


def extract_model_details_from_filename(filename):
    """Extract model name and size details from a filename."""
    split_file_name = filename.split("/")[-1].split("_")
    model_name = split_file_name[0]
    dataset = split_file_name[1]
    filter_type = f"{split_file_name[2]}_{split_file_name[3]}"
    filter_size = split_file_name[4]
    embedding_step = split_file_name[5]
    return model_name,dataset, filter_type, filter_size, embedding_step

def get_filter_info(filename):
    return filename.split("/")[-1].split("_")[-2]

def normalize_per_column(t):
    # Subtract mean of each column from the respective column
    t = t - t.mean(dim=0, keepdim=True)
    # Divide each column by its range
    t = t / (t.max(dim=0, keepdim=True).values - t.min(dim=0, keepdim=True).values)
    return t

def mmd_fake_vs_real(all_files, num_samples=500):
    results = {}
    real_images = [file for file in all_files["0.0"]]
    for real_image in tqdm(real_images):
        X_real_embedding = torch.load(real_image).to(device)[:num_samples, :]
        model_name, dataset, filter_type, filter_size, embedding_step = extract_model_details_from_filename(real_image)
        filtered_embedding = [
            file for file in all_files["1.0"] 
            if model_name in file 
            and filter_size in get_filter_info(file) 
            and embedding_step in file
            and dataset in file][0]
        
        X_fake_embedding = torch.load(filtered_embedding).to(device)[:num_samples, :]
      
        #X_fake_embedding = normalize_per_column(X_fake_embedding)
        #X_real_embedding = normalize_per_column(X_real_embedding)
        if model_name not in results:
            results[model_name] = {}
        if dataset not in results[model_name]:
            results[model_name][dataset] = {}
        if filter_size not in results[model_name][dataset]:
            results[model_name][dataset][filter_size] = {}
        
        results[model_name][dataset][filter_size][embedding_step] = MMD(X_real_embedding, X_fake_embedding).to("cpu").detach().numpy()
    return results

def mmd_no_filter_vs_filter(all_files, num_samples=500):
    results = {}
    for key,value in tqdm(all_files.items()):
        models_results = {}
        path_to_no_filter = [file for file in value if "no" == get_filter_info(file)]

        for real_image in tqdm(path_to_no_filter):
            X_real_embedding = torch.load(real_image).to(device)[:num_samples, :]
            model_name, filter_type, filter_size, embedding_step = extract_model_details_from_filename(real_image)
            #Get the fake images for the same model 
            filtered_embeddings = [file for file in value if model_name in file and "no" != get_filter_info(file) and embedding_step in file]#and "0.0"  not in file
            mmds_DM = {}
            
            for filtered_embedding in filtered_embeddings:
                new_filter_size = get_filter_info(filtered_embedding)
                X_fake_embedding = torch.load(filtered_embedding).to(device)[:num_samples, :]
                mmds_DM[new_filter_size] = MMD(X_real_embedding, X_fake_embedding).to("cpu").detach().numpy()
            if model_name not in models_results:
                models_results[model_name] = {}
            if filter_size not in models_results[model_name]:
                models_results[model_name][filter_size] = {}
            models_results[model_name][filter_size][embedding_step] = mmds_DM
        results[key] = models_results
            
    return results


def build_MMD(input_path, output_path,experiemnt,num_samples=500):
    
    all_files = get_files_from_sub_directory(input_path)
    for experiment_name, files in all_files.items():
        if experiemnt == "fake_vs_real":
            results = mmd_fake_vs_real(files,num_samples)
        elif experiemnt == "no_filter_vs_filter":
            results = mmd_no_filter_vs_filter(files,num_samples)

        new_path = os.path.join(output_path, "mmd_features",f"{experiemnt_name}")
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        output_filename = os.path.join(new_path, f"{experiment_name}.pkl")
        with open(output_filename, "wb") as f:
            pickle.dump(results, f)
    # real_images_no_filter = [file for file in all_files["0.0"]]
    # for real_image in tqdm(real_images_no_filter):
    #     X_real_embedding = torch.load(real_image).to(device)[:num_samples, :]
    #     model_name, model_size = extract_model_details_from_filename(real_image)
    #     real_filter_info = get_filter_info(real_image)
    #     filtered_embeddings = [file for file in all_files["1.0"] if model_name in file and model_size in file]
    #     mmds_DM = {}
    #     for filtered_embedding in filtered_embeddings:
    #         filter_size = get_filter_info(filtered_embedding)
    #         X_fake_embedding = torch.load(filtered_embedding).to(device)[:num_samples, :]
    #         mmds_DM[filter_size] = MMD(X_real_embedding, X_fake_embedding).to("cpu").detach().numpy()
    #     new_path = os.path.join(output_path, "mmd_features",f"{experiemnt_name}")
    #     if not os.path.exists(new_path):
    #         os.makedirs(new_path)
    #     output_filename = os.path.join(new_path, f"{real_filter_info}_{model_name}_{model_size}.pkl")
    #     with open(output_filename, "wb") as f:
    #         pickle.dump(mmds_DM, f)
    
    # for key, value in all_files.items():
    #     path_to_no_filter = [file for file in value if "real" in file]
    # real_images = [file for file in all_files["0.0"]]
    # for real_image in tqdm(real_images):
    #     X_real_embedding = torch.load(real_image).to(device)[:num_samples, :]
    #     model_name, model_size = extract_model_details_from_filename(real_image)
    #     real_filter_info = get_filter_info(real_image)
    #     filtered_embeddings = [file for file in all_files["1.0"] if model_name in file and real_filter_info in get_filter_info(file)]
    #     mmds_DM = {}
    #     for filtered_embedding in filtered_embeddings:
    #         filter_size = get_filter_info(filtered_embedding)
    #         X_fake_embedding = torch.load(filtered_embedding).to(device)[:num_samples, :]
    #         mmds_DM[filter_size] = MMD(X_real_embedding, X_fake_embedding).to("cpu").detach().numpy()
    #     new_path = os.path.join(output_path, "mmd_features",f"{experiemnt_name}",f"{source_data}")
    #     if not os.path.exists(new_path):
    #         os.makedirs(new_path)
    #     output_filename = os.path.join(new_path, f"{real_filter_info}_{model_name}_{model_size}.pkl")
    #     with open(output_filename, "wb") as f:
    #         pickle.dump(mmds_DM, f)
    
    # for key,value in tqdm(all_files.items()):

    #     path_to_no_filter = [file for file in value if "real" in file]
    #     for real_image in tqdm(path_to_no_filter):
    #         X_real_embedding = torch.load(real_image).to(device)[:num_samples, :]
    #         model_name, model_size = extract_model_details_from_filename(real_image)
    #         real_filter_info = get_filter_info(real_image)
    #         #Get the fake images for the same model 
    #         filtered_embeddings = [file for file in value if model_name in file and "real" not in file ]#and "0.0"  not in file
    #         mmds_DM = {}
    #         for filtered_embedding in filtered_embeddings:
    #             filter_size = get_filter_info(filtered_embedding)
    #             X_fake_embedding = torch.load(filtered_embedding).to(device)[:num_samples, :]
    #             mmds_DM[filter_size] = MMD(X_real_embedding, X_fake_embedding).to("cpu").detach().numpy()
    #         new_path = os.path.join(output_path, "mmd_features",f"{experiemnt_name}")
    #         if not os.path.exists(new_path):
    #             os.makedirs(new_path)
    #         output_filename = os.path.join(new_path, f"{key}_{model_name}_{real_filter_info}.pkl")
    #         with open(output_filename, "wb") as f:
    #             pickle.dump(mmds_DM, f)
    


def build_tsne(input_path, output_path,experiemnt_name, num_samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_files = get_files_from_directory(input_path)
    
    real_images_no_filter = [file for file in all_files["0.0"]]
    for real_image_no_filter in tqdm(real_images_no_filter):
        X_real_image_not_filter = torch.load(real_image_no_filter).to(device)[:num_samples, :]
        model_name, model_size = extract_model_details_from_filename(real_image_no_filter)
        real_not_filter = get_filter_info(real_image_no_filter)
        fake_images_with_filter = [file for file in all_files["1.0"] if model_name in file and real_not_filter in get_filter_info(file) ]
        tsne_embedding = {}
        for fake_image_with_filter in fake_images_with_filter:
            filter_size = get_filter_info(fake_image_with_filter)
            X_fake_image_filter = torch.load(fake_image_with_filter).to(device)[:num_samples, :]
            Xs = torch.cat([X_fake_image_filter,X_real_image_not_filter], dim=0).to("cpu").numpy()
            tsne_embedding[filter_size] = TSNE(n_components=2, init='random', perplexity=3, verbose=True).fit_transform(Xs)

        path = os.path.join(output_path, "t-sne",f"{experiemnt_name}")
        if not os.path.exists(path):
            os.makedirs(path)
        output_filename = os.path.join(path, f"{model_name}_{real_not_filter}.pkl")
        with open(output_filename, "wb") as f:
            pickle.dump(tsne_embedding, f)
        

def build(input_path, output_path,experiemnt_name, mode="mmd", num_samples=500):
    """Wrapper function that chooses either MMD or t-SNE based on the mode parameter."""
    if mode == "mmd":
        build_MMD(input_path, output_path,experiemnt_name, num_samples=num_samples)
    elif mode == "tsne":
        build_tsne(input_path, output_path,experiemnt_name, num_samples=num_samples)
    else:
        print(f"Unknown mode: {mode}. Available modes are 'mmd' and 'tsne'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load pre-computed representations and compute MMD or t-SNE visualizations.")
    num_samples = 4000
    parser.add_argument("--base_input_folder", "-b", type=str, default=DEFAULT_BASE_INPUT_FOLDER, help="Base input folder containing feature representations")
    parser.add_argument("--mode","-m", type=str, default="mmd", choices=["mmd", "tsne"], help="Choose between 'mmd' and 'tsne' mode.")
    parser.add_argument("--experiment","-e", type=str, default="experiment", help="Name of the experiment.")
    #parser.add_argument("--input_data","-d", type=str, default="", help="Name of the experiment.")
    parser.add_argument("--output_folder", "-o", type=str, default=FEATURE_DIR, help="Folder to save the MMD results to.")
    parser.add_argument("--sample_size", "-s", type=int, default=num_samples, help="Number of samples to use for MMD computation")
    config = parser.parse_args()
    base_input_folder = config.base_input_folder
    #base_input_folder = os.path.join(base_input_folder,config.input_data)
    print(base_input_folder)
    mode = config.mode
    experiemnt_name = config.experiment
    
    build(base_input_folder, config.output_folder,experiemnt_name, mode=mode,num_samples=config.sample_size)
