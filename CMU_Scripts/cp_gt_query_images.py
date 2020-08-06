import glob
import os
import subprocess
import sys
import numpy as np

#This file is to create sessions folder but only for query images that I can get GT poses for.
# /home/alex/fullpipeline/colmap_data/CMU_data/slice9/live/images/ make sure is empty first

query_images_gt_poses_dir = sys.argv[1] #/home/alex/Datasets/Extended-CMU-Seasons/Extended-CMU-Seasons/slice9/camera-poses/
images_dest = sys.argv[2] #/home/alex/fullpipeline/colmap_data/CMU_data/slice9/live/images/

os.chdir(query_images_gt_poses_dir)
i=0
for file in glob.glob("*.txt"):
    i+=1
    session_folder = images_dest+"session_"+str(i)
    subprocess.run(["mkdir", session_folder ])
    with open(file) as f:
        lines = f.readlines()
    images = []
    for line in lines:
        image_name = line.split(" ")[0]
        if (image_name.split('_')[2] == 'c0'):
            images.append(image_name)

    images = np.array(images)
    images = images[0::4]
    for image_name in images:
        subprocess.run(["cp", "../query/"+image_name, session_folder])

