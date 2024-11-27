import os
import gzip
import numpy as np

# Get the directory containing the current module
FILEDIR = os.path.dirname(os.path.abspath(__file__))

def load(dataset='digits', split='train'):
    """
    Load one of the MNIST-style datasets.

    Args:
        dataset (str): The dataset to load. Must be one of 'digits', 'fashion', 'letters', 'super'.
        split (str): The split to load. Must be one of 'train', 'test'.

    Returns:
        tuple: A tuple containing the images and labels.
    """
    assert dataset in ['digits', 'fashion', 'letters', 'super'], \
        "Dataset must be one of 'digits', 'fashion', 'letters', 'super'"
    
    assert split in ['train', 'test'], \
        "Split must be one of 'train', 'test'"
    
    labels_path = os.path.join(
        FILEDIR, dataset, f"{dataset}-{split}-labels-idx1-ubyte.gz"
    )

    images_path = os.path.join(
        FILEDIR, dataset, f"{dataset}-{split}-images-idx3-ubyte.gz"
    )

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(
            lbpath.read(), dtype=np.uint8, offset=8
        )

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16
        ).reshape(len(labels), 784)

    return images, labels


if __name__ == "__main__":
    imgs, lbs = load(dataset='digits', split='train')
    print(imgs.shape, lbs.shape)