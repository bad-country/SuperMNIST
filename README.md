# SuperMNIST

SuperMNIST is an augmented version of the MNIST dataset that includes transformed versions of the original handwritten digits, effectively increasing the size of the training set.

## Dataset Description

The dataset consists of:
- Original MNIST images (28x28 grayscale)
- Transformed versions including:
  - Inverted images (black ↔ white)
  - [Add other transformations you've implemented]

### Dataset Size
- Original MNIST Training Set: 60,000 images
- SuperMNIST Training Set: 360,000 images (6x original)
- Test Set: [Your test set size]

## Installation

```bash
Clone the repository
git clone https://github.com/yourusername/SuperMNIST.git
cd SuperMNIST
Create and activate conda environment
conda create -n ml-research python=3.12
conda activate ml-research
Install dependencies
pip install numpy matplotlib
``` 

## Usage

```bash
python
from utils import load_mnist, create_super_mnist
Load the original MNIST dataset
train_imgs, train_lbs, test_imgs, test_lbs = load_mnist()
Create the augmented SuperMNIST dataset
super_train_images, super_train_labels, super_test_images, super_test_labels = create_super_mnist()
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