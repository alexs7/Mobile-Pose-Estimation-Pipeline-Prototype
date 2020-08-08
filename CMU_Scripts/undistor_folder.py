import glob
import os
import cv2
import numpy as np
import sys

image_dir = sys.argv[1] # trailing /
folder_nested = sys.argv[2]

def undistort(img):
    distortion_params = np.array([-0.399431, 0.188924, 0.000153, 0.000571])
    fx = 868.993378
    fy = 866.063001
    cx = 525.942323
    cy = 420.042529
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    undst = cv2.undistort(img, K, distortion_params, None, K)
    return undst

if(folder_nested == '1'):
    os.chdir(image_dir)
    for folder in glob.glob("*"):
        for image in glob.glob(folder + "/*.jpg"):
            img = cv2.imread(image)
            undistorted_img = undistort(img)
            cv2.imwrite(image, undistorted_img)
else:
    os.chdir(image_dir)
    for file in glob.glob("*.jpg"):
        img = cv2.imread(file)
        undistorted_img = undistort(img)
        cv2.imwrite(image_dir + file, undistorted_img)