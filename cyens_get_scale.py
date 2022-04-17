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
    model_images_names = get_images_names(model_images)
    model_images_names = model_images_names[:10] # TODO: Only use the ones from Unity (Will need to change that in the future)
    unity_cam_centers = {} #This in metric

    for file in glob.glob(os.path.join(unity_devices_poses_path,"*.txt")):
        with open(file) as f:
            lines = f.readlines()
        values = lines[0].split(',')
        tx = float(values[0])
        ty = float(values[1])
        tz = float(values[2])
        qx = float(values[3])
        qy = float(values[4])
        qz = float(values[5])
        qw = float(values[6])
        cam_center = np.array([tx, ty, tz]) # In Unity the matrices' t component is the camera center in the world
        unity_cam_centers["frame_"+file.split("_")[-1].split(".")[0]+".jpg"] = cam_center

    scales = []
    for i in range(5000): #just to be safe
        random_images = random.sample(model_images_names, 2)

        unity_1_center = unity_cam_centers[random_images[0]]
        unity_2_center = unity_cam_centers[random_images[1]]

        model_cntr1 = get_image_camera_center_by_name(random_images[0], model_images)
        model_cntr2 = get_image_camera_center_by_name(random_images[1], model_images)

        model_cam_dst = np.linalg.norm(model_cntr1 - model_cntr2)
        unity_cam_dst = np.linalg.norm(unity_1_center - unity_2_center) #in meters

        scale = unity_cam_dst / model_cam_dst
        scales.append(scale)

    scale = np.mean(scales)
    print("Scale: " + str(scale))
    return scale

unity_poses_path = sys.argv[1]
colmap_poses_path = sys.argv[2]
calc_scale_COLMAP_UNITY(unity_poses_path,colmap_poses_path)
