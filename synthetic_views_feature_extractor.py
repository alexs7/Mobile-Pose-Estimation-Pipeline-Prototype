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
image_path = os.path.join(base_path, "images/")
image_features_path = os.path.join(base_path, "images_features/")

if not os.path.exists(image_features_path):
    os.makedirs(image_features_path)

for filename in os.scandir(image_path):
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