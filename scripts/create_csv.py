import argparse
import csv
import glob
from typing import List
from math import floor
from sklearn.model_selection import train_test_split
import os
def write_to_csv(file_path: str, data: List[tuple[str, int]]) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(data)

def read_paths_recursive(path: str, label: int) -> List[tuple[str, int]]:
    edited_videos = glob.glob(path, recursive=True)
    return [(edit, label) for edit in edited_videos]

def oversample_minority_class(data: List[tuple[str, int]]):
    # Separate data based on labels
    class_0 = [item for item in data if item[1] == 0]
    class_1 = [item for item in data if item[1] == 1]

    # Determine which class is the minority
    if len(class_0) < len(class_1):
        minority_class, majority_class = class_0, class_1
    else:
        minority_class, majority_class = class_1, class_0

    # Oversample the minority class by duplicating its samples
    multiplier = len(majority_class) // len(minority_class)
    oversampled_minority = minority_class * multiplier

    # Combine the oversampled minority class with the majority class
    combined_data = oversampled_minority + majority_class

    return combined_data

def undersmaple_majority_class(data: List[tuple[str, int]]):
    # Separate data based on labels
    class_0 = [item for item in data if item[1] == 0]
    class_1 = [item for item in data if item[1] == 1]

    # Determine which class is the minority
    if len(class_0) < len(class_1):
        minority_class, majority_class = class_0, class_1
    else:
        minority_class, majority_class = class_1, class_0

    # Oversample the minority class by duplicating its samples
    multiplier = len(majority_class) // len(minority_class)
    
    undersmapled_majority = [majority_class[floor(i*multiplier)] for i in range(len(minority_class))]
    # Combine the oversampled minority class with the majority class
    combined_data = undersmapled_majority + minority_class

    return combined_data

def split_data(data, train_ratio=0.80, val_ratio=0.05):
    train, test = train_test_split(
        data, test_size=1 - train_ratio, random_state=42, shuffle=True
    )
    val, test = train_test_split(
        test, test_size=val_ratio / (1.0 - train_ratio), random_state=42, shuffle=True
    )
    return train, val, test
def main(args, data_type):
    orginal_data = args.original_data_path
    base_path = args.base_path
    
    edited_data_path= os.path.join(args.edited_data_path, data_type)
    orginal_data = os.path.join(orginal_data, data_type)
    orginal = read_paths_recursive(f"{orginal_data}/0_real{args.read_key}", 0)
    edited = read_paths_recursive(f"{edited_data_path}/1_fake{args.read_key}", 1)

    combined = edited + orginal
    print("Start Sampling")
    print(len(combined))

    data = oversample_minority_class(combined)

    print("Start Splitting Data")
    train_data, val_data, test_data = split_data(data, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    print(f"Len Train Data: {len(train_data)}")
    write_to_csv(base_path + f"train_data_{data_type}.csv", train_data)
    write_to_csv(base_path + f"val_data_{data_type}.csv", val_data)
    write_to_csv(base_path + f"test_data_{data_type}.csv", test_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI tool for data processing.')
    parser.add_argument('--original_data_path', type=str, required=True, help='Path to the original data')
    parser.add_argument('--edited_data_path', type=str, required=True, help='Path to the edited data')
    parser.add_argument('--base_path', type=str, required=True, help='Base path for output files')
    parser.add_argument('--data_type', type=str, default='coco', help='Data type (default: coco)')
    parser.add_argument('--read_key', type=str, default='coco', help='Key to read data (default: coco)')
    parser.add_argument('--for_all', action='store_true', help='Flag to process all data types')
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Ratio of validation data")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Ratio of test data")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of train data")

    args = parser.parse_args()

    # List of data types
    data_types = ["ADM", "DDPM", "Diff-ProjectedGAN", "Diff-StyleGAN2", "IDDPM", "LDM", "PNDM", "ProGAN", "ProjectedGAN", "StyleGAN"]

    if args.for_all:
        for data_type in data_types:
            main(args, data_type)
    else:
        main(args, args.data_type)

