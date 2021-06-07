# from PIL import Image
import cv2 as cv
import numpy as np
import os


imgDir = "E://OPM Colorings//downloads//"

# for every image in the training set
for img in os.listdir(imgDir):
    # get a image array
    image = np.array(cv.imread(imgDir + img))
