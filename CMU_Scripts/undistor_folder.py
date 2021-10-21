import glob
import os
import shutil

import cv2
import numpy as np
import sys

image_dir = sys.argv[1] # trailing /
output_folder = os.path.join(image_dir, "undistorted")

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
else:
    shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

def undistort(img):
    k1 = -0.0736670151808704
    k2 = 0.0236802942880179
    k3 = -0.0139645232036635
    p1 = -0.00783895854471095
    p2 = 0.00916247523710901
    distortion_params = np.array([k1, k2, p1, p2, k3])
    fx = 3427.46123770218
    fy = 3424.05295846601
    cx = 2828.29293947945
    cy = 1420.23232712147
    skew = 0
    K = np.array([[fx, skew, cx], [0, fy, cy], [0, 0, 1]])
    undst = cv2.undistort(img, K, distortion_params, None, K)
    return undst

images_full_path = os.path.join(image_dir, "*.JPG")

for file in glob.glob(images_full_path):
    print("Undistorting: " + file)
    img = cv2.imread(file)
    undistorted_img = undistort(img)
    output_file = os.path.join(output_folder,file.split("/")[-1])
    cv2.imwrite(output_file, undistorted_img)