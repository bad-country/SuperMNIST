# SuperMNIST

SuperMNIST is an augmented version of the MNIST, FashionMNIST, and NotMNIST datasets that includes transformed versions of the original images, effectively increasing the size of the training set and the number of classes.  It is larger and more difficult than the original datasets, yet still small enough to be used for simple machine learning experiments on a laptop.

## Dataset Description

This repo includes four datasets:

- MNIST: 60,000 training images and 10,000 test images of handwritten digits (28x28 grayscale)
- FashionMNIST: 60,000 training images and 10,000 test images of fashion items (28x28 grayscale)
- NotMNIST: 60,000 training images and 10,000 test images of characters A-J (28x28 grayscale)
- SuperMNIST: 360,000 training images and 10,000 test images of transformed MNIST digits (28x28 grayscale)

MNIST, FashionMNIST, and NotMNIST all contain 10 classes, while SuperMNIST contains 30 classes.

SuperMNIST is created by combining images from MNIST, FashionMNIST, and NotMNIST along with their color flipped versions. 

## Performance

The following are the validation accuracies for the MNIST, FashionMNIST, and NotMNIST datasets using a simple 3-layer MLP classifier with 128 hidden units.

MNIST: Validation accuracy = 94%
FashionMNIST: Validation accuracy = 85%
NotMNIST: Validation accuracy = 91%
SuperMNIST: Validation accuracy = 90%

## Installation

```bash
# Clone the repository
git clone https://github.com/drckf/SuperMNIST.git
cd SuperMNIST
pip install -e .
``` 

## Usage

```bash
python
from utils import load_mnist, create_super_mnist
Load the original MNIST dataset
train_imgs, train_lbs = load_mnist(dataset='digits', split='train')
test_imgs, test_lbs = load_mnist(dataset='digits', split='test')
Create the augmented SuperMNIST dataset
create_super_mnist()
```

## File Structure

```
SuperMNIST/
├── data/
│   ├── digits/
│   ├── letters/
│   ├── fashion/
│   ├── super/
├── utils.py
``` 

## Data Files
Note: Large data files are stored using Git LFS. Make sure you have Git LFS installed to properly clone the repository.

## License
MIT License

Copyright (c) 2024 Charles K. Fisher.

See LICENSE.txt for full license text.

## Acknowledgments

Please cite the following papers if you use this dataset:

```
@article{lecun2010mnist,
  title={MNIST handwritten digit database},
  author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
  journal={ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist},
  volume={2},
  year={2010}
}

@online{xiao2017/online,
  author       = {Han Xiao and Kashif Rasul and Roland Vollgraf},
  title        = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  date         = {2017-08-28},
  year         = {2017},
  eprintclass  = {cs.LG},
  eprinttype   = {arXiv},
  eprint       = {cs.LG/1708.07747},
}

@misc{bulatov2011notmnist,
  author = {Yaroslav Bulatov},
  title = {notMNIST dataset},
  year = {2011},
  month = sep,
  url = {https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html},
}
```