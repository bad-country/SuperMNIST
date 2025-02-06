# MIT License
#
# Copyright (C) 2025 Bad Country LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm

# Get the directory containing the current module
FILEDIR = os.path.dirname(os.path.abspath(__file__))


def load(dataset='digits', split='train', normalize=False, sorted=False):
    """
    Load one of the MNIST-style datasets.

    digits: MNIST
    fashion: FashionMNIST
    letters: NotMNIST
    super: SuperMNIST

    Args:
        dataset (str): The dataset to load. Must be one of 'digits', 'fashion', 'letters', 'super'.
        split (str): The split to load. Must be one of 'train', 'test'.
        normalize (bool, optional): Whether to normalize the images to [0,1]. Defaults to False.
        sorted (bool, optional): Whether to sort the images by label. Defaults to False.

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

    if normalize:
        images = images.astype(np.float32) / 255.0

    if sorted:
        sorted_indices = np.argsort(labels)
        images = images[sorted_indices]
        labels = labels[sorted_indices]

    return images, labels


def save_images(data, filename):
    """Save images in MNIST ubyte format."""
    with gzip.open(filename, 'wb') as f:
        # Write MNIST magic number for images
        f.write(np.array([0x803, len(data), 28, 28], dtype='>i4').tobytes())
        # Write images
        f.write(data.astype('>u1').tobytes())


def save_labels(data, filename):
    """Save labels in MNIST ubyte format."""
    with gzip.open(filename, 'wb') as f:
        # Write MNIST magic number for labels
        f.write(np.array([0x801, len(data)], dtype='>i4').tobytes())
        # Write labels            
        f.write(data.astype('>u1').tobytes())


def create_super_mnist(force=False):

    # paths to output files
    super_train_labels_path = os.path.join(FILEDIR, "super", "super-train-labels-idx1-ubyte.gz")
    super_train_images_path = os.path.join(FILEDIR, "super", "super-train-images-idx3-ubyte.gz")
    super_test_labels_path = os.path.join(FILEDIR, "super", "super-test-labels-idx1-ubyte.gz")
    super_test_images_path = os.path.join(FILEDIR, "super", "super-test-images-idx3-ubyte.gz")

    # check if datasets already exist
    if force == False:
        check = any(
            (
                os.path.exists(super_train_labels_path),
                os.path.exists(super_train_images_path),
                os.path.exists(super_test_labels_path),
                os.path.exists(super_test_images_path)
            )
        )
        if check:
            print("Super MNIST already exists. Set force=True to overwrite.")
            return

    super_train_images = np.empty((0,784), dtype=np.uint8)
    super_train_labels = np.empty((0,), dtype=np.uint8)
    super_test_images = np.empty((0,784), dtype=np.uint8)
    super_test_labels = np.empty((0,), dtype=np.uint8)

    class_offset = 0
    for dataset in ["digits", "letters", "fashion"]:
        train_imgs, train_lbs = load(dataset=dataset, split='train')
        test_imgs, test_lbs = load(dataset=dataset, split='test')

        # concatenate the data
        super_train_images = np.concatenate((super_train_images, train_imgs))
        super_train_labels = np.concatenate((super_train_labels, class_offset + train_lbs))
        super_test_images = np.concatenate((super_test_images, test_imgs))
        super_test_labels = np.concatenate((super_test_labels, class_offset + test_lbs))

        # add the flipped images
        super_train_images = np.concatenate((super_train_images, 255 - train_imgs))
        super_train_labels = np.concatenate((super_train_labels, class_offset + train_lbs))
        super_test_images = np.concatenate((super_test_images, 255 - test_imgs))
        super_test_labels = np.concatenate((super_test_labels, class_offset + test_lbs))

        class_offset += 10

    # save the data
    save_images(super_train_images, super_train_images_path)
    save_labels(super_train_labels, super_train_labels_path)
    save_images(super_test_images, super_test_images_path)
    save_labels(super_test_labels, super_test_labels_path) 


def plot_image(
        image_vector, 
        shape, 
        vmin=0, 
        vmax=255, 
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
        vmax (float, optional): Defaults to 255.
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
        vmax=255, 
        filename=None, 
        show=True,
        cmap=cm.gray
    ):
    """
    Plot a grid of images.

    Args:
        image_array (np.ndarray):
        shape (tuple(int, int)):
        vmin (float, optional): Defaults to 0.
        vmax (float, optional): Defaults to 255.
        filename (str, optional): Defaults to None.
        show (bool, optional): Defaults to True.
        cmap (optional): Defaults to cm.gray.
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
    # example usage
    np.random.seed(42)
    create_super_mnist(force=True)
    imgs, lbs = load(dataset='super', split='train', normalize=True, sorted=True)
    print(imgs.shape, lbs.shape)
    plot_image(imgs[0], (28, 28), vmin=0, vmax=1)
    plot_image_grid(imgs[:6].reshape(2,3,784), (28, 28), vmin=0, vmax=1)
    random_indices = np.random.choice(len(imgs), size=6, replace=False)
    plot_image_grid(imgs[random_indices].reshape(2,3,784), (28, 28), vmin=0, vmax=1)
    print(lbs[random_indices])
