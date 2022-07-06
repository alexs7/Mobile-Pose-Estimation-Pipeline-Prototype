import glob
import sys
import time
from os.path import join
import numpy as np
import open3d as o3d
import os
import cv2

# This file will be used to create a database of feature descriptors

WIDTH = 1920
HEIGHT = 1080

start = time.time()

base_path = sys.argv[1] # i.e. /Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/
synth_images_path = os.path.join(base_path, "synth_images/")
depths_path = os.path.join(base_path, "depths/")
synth_image_features_path = os.path.join(base_path, "images_features/")
no_images = len(glob.glob(os.path.join(synth_images_path, "*.png")))

sift = cv2.SIFT_create()

if not os.path.exists(synth_image_features_path):
    os.makedirs(synth_image_features_path)

print("Getting features..")
for i in range(no_images):
    synth_image_path = os.path.join(synth_images_path, "{:05d}.png".format(i))
    depth_path = os.path.join(depths_path, "{:05d}.png".format(i))

    synth_image = cv2.imread(synth_image_path, cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    mask = np.copy(depth)
    mask[np.where(mask > 0)] = 255

    kps, descs = sift.detectAndCompute(synth_image, mask = mask)

    img_x_y = np.empty([0, 2])

    for k in range(len(kps)):
        kp = kps[k]
        x = kp.pt[0]
        y = kp.pt[1]
        img_x_y = np.r_[img_x_y, np.array([x, y]).reshape(1,2)]

    img_data = np.c_[img_x_y, descs]
    np.save(os.path.join(synth_image_features_path, "{:05d}.npy".format(i)), img_data)

print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))

exit()