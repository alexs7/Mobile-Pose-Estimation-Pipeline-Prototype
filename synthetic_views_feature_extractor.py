import glob
import sys
import time
from os.path import join
import numpy as np
import open3d as o3d
import os
import cv2

WIDTH = 1920
HEIGHT = 1080

base_path = sys.argv[1] # i.e. /Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/
images_path = os.path.join(base_path, "images/")
depths_path = os.path.join(base_path, "depths/")
image_features_path = os.path.join(base_path, "images_features/")
no_images = len(glob.glob(os.path.join(images_path,"*.png")))

if not os.path.exists(image_features_path):
    os.makedirs(image_features_path)

for i in range(len(no_images)):
    breakpoint()

for filename in os.scandir(depths_path):
    if filename.is_file():
        print(filename.path)
        breakpoint()
        cv2.imread(filename.path, cv2.IMREAD_GRAYSCALE)

for filename in os.scandir(images_path):
    if filename.is_file():
        print(filename.path)
        breakpoint()
        cv2.imread(filename.path, cv2.IMREAD_GRAYSCALE)

start = time.time()


print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))

exit()