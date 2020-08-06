import glob
import os
import sys
import numpy as np
from query_image import get_image_camera_center_by_name, read_images_binary

# /home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/live
# /home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/gt

# TODO: might need to refactor this

# this will generate the text file needed for model aligner
# the resulting file will contain base images names with their corresponding reference model centers
def gen_base_cam_centers_txt(base_images_dir, reference_model_images_path):
    images = []
    base_images = []

    for file in glob.glob(base_images_dir+"/*.jpg"):
        name = file.split("/")[-1]
        base_images.append(name)

    with open(reference_model_images_path) as f:
        lines = f.readlines()

    for line in lines:
        image_name = line.split(" ")[0]
        if(image_name in base_images):
            x = line.split(" ")[-4]
            y = line.split(" ")[-3]
            z = line.split(" ")[-2]
            images.append(image_name + " " + x + " " + y + " " + z)

    with open(base_images_dir+'/../base_images_cam_centers.txt', 'w') as f:
        for image in images:
            f.write("%s\n" % image)

def gen_query_txt(dir, base_images_no = None):
    session_nums = []
    images = []
    for folder in glob.glob(dir+"/session_*"):
        i=0
        for file in glob.glob(folder+"/*.jpg"):
            image_name = file.split('images/')[1]
            images.append(image_name)
            i+=1
        session_nums.append(i)

    with open(dir+'/../query_name.txt', 'w') as f:
        for image in images:
            f.write("%s\n" % image)

    session_nums =  [base_images_no] + session_nums
    if(base_images_no is not None) : np.savetxt(dir+"/../session_lengths.txt", session_nums)
