
import os

def ensure_dir(path):
    """Ensure that the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def create_path(*args):
    """Create a path from the arguments."""
    return os.path.join(*args)