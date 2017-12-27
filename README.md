# Panorama
This repository includes the following files.

--- FeatureMatching.m

Acknowledgement: 
this code used SIFT feature matching with the VLFeat open source library, available at http://www.vlfeat.org/overview/sift.html.

parameter: 
image1, image2

output: 
randomly selected 350 matching features between image1 and image2



--- Panorama.m
This code combines 5 pictures to make a panorama of them.

Acknowledgement: 
this code used SIFT feature matching with the VLFeat open source library, available at http://www.vlfeat.org/overview/sift.html and some guidelines for picture warping, available at https://www.mathworks.com/help/vision/examples/feature-based-panoramic-image-stitching.html.

parameter: 
folder that contains the source pictures to create panorama with
please note that the pictures need to be named as numbers with their panoramic order from let to right

output: 
panorama picture (you will need further cropping for a rectangular picture result)

--- Dataset1
This folder includes source pictures to create panorama with.