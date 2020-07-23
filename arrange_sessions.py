import glob
import os
import sys
import numpy as np
from query_image import get_image_camera_center_by_name, read_images_binary

# /home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/live
# /home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/gt

# TODO: might need to refactor this

# this will generate the text file needed for model aligner
def gen_base_cam_centers_txt(base_images_dir, reference_model_images_path):
    session_nums = []
    images = []
    base_model_images = read_images_binary(reference_model_images_path)

    for file in glob.glob(base_images_dir+"/*.jpg"):
        image_name = file.split("/")[-1]
        camera_center = get_image_camera_center_by_name(image_name, base_model_images) # assume all images are localised in the reference model
        data = image_name + " " + str(camera_center[0]) + " " + str(camera_center[1]) + " " + str(camera_center[2]) + "\n"
        images.append(data)

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
