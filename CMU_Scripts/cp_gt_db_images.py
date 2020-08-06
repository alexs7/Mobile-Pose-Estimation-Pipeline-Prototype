import glob
import os
import subprocess
import sys
import numpy as np

#This file is to create sessions folder but only for database (for SFM) images that I can get GT poses for.
# /home/alex/fullpipeline/colmap_data/CMU_data/slice9/base/images/ make sure is empty first

db_images_gt_poses_dir = sys.argv[1] #/home/alex/Datasets/Extended-CMU-Seasons/Extended-CMU-Seasons/slice9/database/
images_dest = sys.argv[2] #/home/alex/fullpipeline/colmap_data/CMU_data/slice9/base/images/

os.chdir(db_images_gt_poses_dir)
images = []
for file in glob.glob("*.jpg"):
    if(file.split('_')[2] == 'c0'):
        images.append(file)

images = np.array(images)
# images = images[0::2]
for image_name in images:
    subprocess.run(["cp", image_name, images_dest])

