#!/usr/bin/env bash


# download files.

wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/README.txt
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/Ranges.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/aeroplane.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/bird.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/boat.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/car.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/cat.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/cow.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/dog.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/horse.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/motorbike.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/train.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/GroundTruth.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/OpticalFlow.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/SlicSuperpixels.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/UsefulFiles.tar.gz
wget -nc http://calvin-vision.net/bigstuff/youtube-objectsv2/videos/YouTubeObjectsVideos.tar.gz



# untar files.

for file in `ls -1 *.tar.gz`; do 
	tar -xf $file &
done
