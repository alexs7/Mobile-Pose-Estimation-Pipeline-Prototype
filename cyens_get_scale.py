# This file is just to get scales between COLMAP and ARCORE, or COLMAP and COLMAP.
# manual work to set the directories
import glob
import sys
import os

import numpy as np
import random
from query_image import read_images_binary, get_image_camera_center_by_name, get_images_names
from scipy.spatial.transform import Rotation as R

def calc_scale_COLMAP_UNITY(unity_devices_poses_path, colmap_model_images_path):

    model_images = read_images_binary(colmap_model_images_path)
    unity_cam_centers = {} #This in metric

    for file in glob.glob(os.path.join(unity_devices_poses_path,"local_pose_*.txt")): #or world_pose is the same
        with open(file) as f:
            lines = f.readlines()
        values = lines[0].split(',')
        tx = float(values[0])
        ty = float(values[1])
        tz = float(values[2])
        cam_center = np.array([tx, ty, tz]) # In Unity the matrices' t component is the camera center in the world
        unity_cam_centers["frame_"+file.split("_")[-1].split(".")[0]+".jpg"] = cam_center

    scales = []
    model_images_names = list(unity_cam_centers.keys()) #we only need the frames localised from the phone (not the ones used to reconstruct the pointcloud)
    for i in range(5000): #just to be safe
        random_images = random.sample(model_images_names, 2)

        unity_1_center = unity_cam_centers[random_images[0]]
        unity_2_center = unity_cam_centers[random_images[1]]

        model_cntr1 = get_image_camera_center_by_name(random_images[0], model_images)
        model_cntr2 = get_image_camera_center_by_name(random_images[1], model_images)

        if(model_cntr1.size==0 or model_cntr2.size==0): #this is to check if the unity frame has been localised or not (i.e is in the model)
            continue

        model_cam_dst = np.linalg.norm(model_cntr1 - model_cntr2)
        unity_cam_dst = np.linalg.norm(unity_1_center - unity_2_center) #in meters

        scale = unity_cam_dst / model_cam_dst
        scales.append(scale)

    scale = np.mean(scales)
    print("Scale: " + str(scale))
    return scale

base_path = sys.argv[1]
unity_poses_path = os.path.join(base_path,"scale_data")
colmap_poses_path = os.path.join(base_path,"new_model/images.bin")
scale_txt_path = os.path.join(base_path,"scale.txt")
scale = calc_scale_COLMAP_UNITY(unity_poses_path,colmap_poses_path)

with open(scale_txt_path, 'w') as f:
    f.write(str(scale))