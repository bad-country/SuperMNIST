# Copyright (C) 2025 Bad Country LLC
#
# This file is part of MLOrchard.
#
# MLOrchard is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# MLOrchard is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with MLOrchard.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="SuperMNIST", 
    version="0.1.0",
    author="Charles K. Fisher",
    author_email="drckf@badcountry.ai",
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