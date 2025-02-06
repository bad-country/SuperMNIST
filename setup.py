from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SuperMNIST", 
    version="0.1.0",
    author="Charles K. Fisher",
    author_email="charleskennethfisher@gmail.com",
    description="SuperMNIST is an augmented version of the MNIST dataset with more images and categories.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drckf/SuperMNIST",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "matplotlib",
    ],
    package_data={
        "": ["*/digits/*", "*/fashion/*", "*/letters/*", "*/super/*"],
    },
)