# roads_detection

This repository contains the following notebooks for road detection.

 - 1_Road_Classification_VGG - Classify whether or not an image tile has roads using VGGNet(in Tensorflow) and 256*256*3 images

- 2_Road_Classification_AlexNet - Classify whether or not an image tile has roads using Alexnet(in Keras) and 256*256*3 images

- 3_Road_Classification_ResNet - Classify whether or not an image tile has roads using 14 layer ResNet(in Keras) and 64*64*3 images

- 3_Road_Pixels_ResNet - Classify whether or not a pixel is part of a road using 14 layer ResNet(in Keras) that takes 64*64*3 images and produces a 64*64 mask as outputs. 

The dataset contains 10000 images from the Rotterdam area in Netherlands and the labels are shapefiles containing co-ordinates of roads. The models are in cnn_models/