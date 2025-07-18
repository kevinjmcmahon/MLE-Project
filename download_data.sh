#!/bin/bash

# Create a directory for the raw data
mkdir -p data/raw

# Download the image and annotation tarballs
wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar -O data/raw/images.tar
wget http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar -O data/raw/lists.tar

# Unpack the data
tar -xvf data/raw/images.tar -C data/raw/
tar -xvf data/raw/lists.tar -C data/raw/

echo "Data downloaded and unpacked in data/raw/"