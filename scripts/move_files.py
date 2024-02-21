import os
import shutil


def copy_first_1000_images(source_directory, destination_directory):
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    for subdir in os.listdir(source_directory):
        full_subdir_path = os.path.join(source_directory, subdir)

        if os.path.isdir(full_subdir_path):
            dest_subdir_path = os.path.join(destination_directory, subdir)
            os.makedirs(dest_subdir_path, exist_ok=True)

            count = 0
            for file in os.listdir(full_subdir_path):
                if count >= 1500:
                    break

                full_file_path = os.path.join(full_subdir_path, file)
                if os.path.isfile(full_file_path) and file.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    shutil.copy(full_file_path, dest_subdir_path)
                    count += 1


# Example usage
source_dir = "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/lsun_bedroom/test"
dest_dir = "/home/wustl/Dummy/Wustl/Deepfake/MasterThesis/data/lsun_bedroom/log"
copy_first_1000_images(source_dir, dest_dir)
