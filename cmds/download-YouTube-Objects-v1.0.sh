#!/usr/bin/env bash


# download files.

wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/code.tar.gz
wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/aeroplane.tar.gz
wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/bird.tar.gz
wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/boat.tar.gz
wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/car.tar.gz
wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/cat.tar.gz
wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/cow.tar.gz
wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/dog.tar.gz
wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/horse.tar.gz
wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/motorbike.tar.gz
wget -nc https://data.vision.ee.ethz.ch/cvl/youtube-objects/categories/train.tar.gz


# untar files.

for file in `ls -1 *.tar.gz`; do 
	tar -xf $file &
done
