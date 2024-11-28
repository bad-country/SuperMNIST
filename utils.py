import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm


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


def plot_image(
        image_vector, 
        shape, 
        vmin=0, 
        vmax=1, 
        filename=None, 
        show=True,
        cmap=cm.gray, 
        nan_color='red'
    ):
    """
    Plot a single image. 

    Args:
        image_vector (np.ndarray): images in vectorized format
        shape (tuple(int, int)): shape of each 2D image
        vmin (float, optional): Defaults to 0.
        vmax (float, optional): Defaults to 1.
        filename (str, optional): Defaults to None.
        show (bool, optional): Defaults to True.
        cmap (optional): Defaults to cm.gray.
        nan_color (str, optional): Defaults to 'red'.

    Returns:
        None
    """
    f, ax = plt.subplots(figsize=(4,4))

    # reshape the data and cast to a numpy array
    img = np.reshape(image_vector, shape)

    # make the plot
    ax.imshow(img, interpolation='none', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(yticks=[])
    ax.set(xticks=[])

    if show:
        plt.show()
    if filename is not None:
        f.savefig(filename)
    plt.close(f)


def plot_image_grid(
        image_array, 
        shape, 
        vmin=0, 
        vmax=1, 
        filename=None, 
        show=True,
        cmap=cm.gray, 
        nan_color='red'
    ):
    """
    Plot a grid of images.

    Args:
        image_array (np.ndarray):
        shape (tuple(int, int)):
        vmin (float, optional): Defaults to 0.
        vmax (float, optional): Defaults to 1.
        filename (str, optional): Defaults to None.
        show (bool, optional): Defaults to True.
        cmap (optional): Defaults to cm.gray.
        nan_color (str, optional): Defaults to 'red'.
    """
    # cast to a numpy array
    nrows, ncols = image_array.shape[:-1]

    f = plt.figure(figsize=(2*ncols, 2*nrows))
    grid = gs.GridSpec(nrows, ncols)
    axes = [[plt.subplot(grid[i,j]) for j in range(ncols)] for i in range(nrows)]
    for i in range(nrows):
        for j in range(ncols):
            axes[i][j].imshow(
                np.reshape(image_array[i][j], shape), 
                cmap=cmap,
                interpolation='none', 
                vmin=vmin, 
                vmax=vmax
            )
            axes[i][j].set(yticks=[])
            axes[i][j].set(xticks=[])
    plt.tight_layout(pad=0.5, h_pad=0.2, w_pad=0.2)
    if show:
        plt.show()
    if filename is not None:
        f.savefig(filename)
    plt.close(f)


if __name__ == "__main__":
    imgs, lbs = load(dataset='digits', split='train')
    print(imgs.shape, lbs.shape)

    plot_image(imgs[0], (28, 28))
    plot_image_grid(imgs[:6].reshape(2,3,784), (28, 28))
