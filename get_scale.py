# This file is just to get scales between COLMAP and ARCORE, or COLMAP and COLMAP.
# manual work to set the directories
import glob
import numpy as np
import random
from query_image import read_images_binary, get_image_camera_center_by_name, get_images_names

def calc_scale_COLMAP_ARCORE(arcore_devices_poses_path, colmap_model_images_path):

    model_images = read_images_binary(colmap_model_images_path)
    model_images_names = get_images_names(model_images)
    ar_core_cam_centers = {} #This in metric

    for file in glob.glob(arcore_devices_poses_path+"displayOrientedPose_*.txt"):
        pose = np.loadtxt(file)
        cam_center = np.array(pose[0:3, 3]) # remember in ARCore the matrices' t component is the camera center in the world
        ar_core_cam_centers["frame_"+file.split("_")[-1].split(".")[0]+".jpg"] = cam_center

    scales = []
    for i in range(5000): #just to be safe
        random_images = random.sample(model_images_names, 2)

        arcore_1_center = ar_core_cam_centers[random_images[0]]
        arcore_2_center = ar_core_cam_centers[random_images[1]]

        model_cntr1 = get_image_camera_center_by_name(random_images[0], model_images)
        model_cntr2 = get_image_camera_center_by_name(random_images[1], model_images)

        model_cam_dst = np.linalg.norm(model_cntr1 - model_cntr2)
        arcore_cam_dst = np.linalg.norm(arcore_1_center - arcore_2_center) #in meters

        scale = arcore_cam_dst / model_cam_dst
        scales.append(scale)

    return np.mean(scales)

def calc_scale_COLMAP(path, colmap_model_images_path):
    slice = path.split('/')[-2]
    original_images_path = "/home/alex/Datasets/Extended-CMU-Seasons/Extended-CMU-Seasons/"+slice+"/ground-truth-database-images-"+slice+".txt"
    model_images = read_images_binary(colmap_model_images_path)

    original_images = []
    with open(original_images_path) as f:
        original_images = f.readlines()

    original_cam_centers = {}
    for original_image in original_images:
        name = original_image.split()[0]
        center = original_image.split()[5:8]
        original_cam_centers[name] = center

    model_images_names = []
    for k,v in model_images.items():
        model_images_names.append(v.name)

    scales = []
    for i in range(1000):
        random_images = random.sample(model_images_names, 2)

        model_cntr1 = get_image_camera_center_by_name(random_images[0], model_images)
        model_cntr2 = get_image_camera_center_by_name(random_images[1], model_images)

        original_cntr1 = np.array(original_cam_centers[random_images[0]]).astype(np.float)
        original_cntr2 = np.array(original_cam_centers[random_images[1]]).astype(np.float)

        model_cam_dst = np.linalg.norm(model_cntr1 - model_cntr2)
        original_cam_dst = np.linalg.norm(original_cntr1 - original_cntr2) #in meters

        scale = original_cam_dst / model_cam_dst
        scales.append(scale)

    return np.mean(scales)
