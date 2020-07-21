# This file is just to get scales between COLMAP and ARCORE, or COLMAP and COLMAP.
# manual work to set the directories
import glob
import numpy as np
import random
from query_image import read_images_binary, get_image_camera_center_by_name

def calc_scale_COLMAP_ARCORE(arcore_poses_path, colmap_model_from_arcore_images_path):
    model_images = read_images_binary(colmap_model_from_arcore_images_path)
    model_images_names = get_images_names(model_images)
    frame_cam_centers = {}
    for file in glob.glob(arcore_poses_path+"displayOrientedPose_*.txt"):
        pose = np.loadtxt(file)
        pose_t = np.array(pose[0:3, 3])
        rot = np.array(pose[0:3, 0:3])
        cam_center = -rot.transpose().dot(pose_t)
        frame_cam_centers["frame_"+file.split("_")[-1].split(".")[0]+".jpg"] = pose_t

    scales = []
    for i in range(1000):
        random_images = random.sample(model_images_names, 2)

        arcore_1_center = frame_cam_centers[random_images[0]]
        arcore_2_center = frame_cam_centers[random_images[1]]

        model_cntr1 = get_image_camera_center_by_name(random_images[0], model_images)
        model_cntr2 = get_image_camera_center_by_name(random_images[1], model_images)

        model_cam_dst = np.linalg.norm(model_cntr1 - model_cntr2)
        arcore_cam_dst = np.linalg.norm(arcore_1_center - arcore_2_center)

        scale = arcore_cam_dst / model_cam_dst

        scales.append(scale)

    return np.mean(scales)

def calc_scale_COLMAP(original_images_path, colmap_model_images_path):
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
        original_cam_dst = np.linalg.norm(original_cntr1 - original_cntr2)

        scale = original_cam_dst / model_cam_dst
        scales.append(scale)

    return np.mean(scales)

# print("Scale: ")
# # # These will change manually
# original_images_path = "/home/alex/Datasets/Extended-CMU-Seasons/Extended-CMU-Seasons/slice17/ground-truth-database-images-slice17.txt"
# colmap_model_images_path = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/base/model/0/images.bin"
#
# print(calc_scale_COLMAP(original_images_path, colmap_model_images_path))

# # they have to correspond
# arcore_poses_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/2020-06-17/morning/run_3/data_all/"
# colmap_model_from_arcore_images_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/2020-06-17/morning/run_3/reconstruction/0/images.bin"
#
# print(calc_scale_COLMAP_ARCORE(arcore_poses_path, colmap_model_from_arcore_images_path))
