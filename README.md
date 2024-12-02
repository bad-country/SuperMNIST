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
[Your chosen license]

## Acknowledgments
- Original MNIST dataset by Yann LeCun and Corinna Cortes