import glob
import sys
import time
from os.path import join
import numpy as np
import open3d as o3d
import os
import cv2
from random import *

start = time.time()

base_path = sys.argv[1] # i.e. /Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/
mesh_path = os.path.join(base_path, "model_files/EXPORT_Mesh/model.fbx")
synth_images_path = os.path.join(base_path, "synth_images/")
images_path = os.path.join(base_path, "IMAGES/all/")
depths_path = os.path.join(base_path, "depths/")
poses_path = os.path.join(base_path, "poses/")
synth_image_features_path = os.path.join(base_path, "images_features/")
no_images = len(glob.glob(os.path.join(synth_images_path, "*.png")))

print("Creating featires dir...") # will store 2D - 3D - SIFT in a numpy .npy file for each image
if not os.path.exists(synth_image_features_path):
    os.makedirs(synth_image_features_path)

print("Reading mesh...")
mesh = o3d.io.read_triangle_mesh(mesh_path)

sift = cv2.SIFT_create()
# FLANN_INDEX_KDTREE = 1 #https://docs.opencv.org/3.4.0/dc/d8c/namespacecvflann.html
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)  # or pass empty dictionary
# matcher = cv2.FlannBasedMatcher(index_params, search_params)
# ratio_thresh = 0.7

test_index = randint(0, no_images)

# real_image_path = os.path.join(images_path, "frame{:06}.png".format(test_index))
# real_img = cv2.imread(real_image_path, cv2.IMREAD_GRAYSCALE)

synth_image_path = os.path.join(synth_images_path, "{:05d}.png".format(test_index))
synth_image = cv2.imread(synth_image_path, cv2.IMREAD_GRAYSCALE)

depth_path = os.path.join(depths_path, "{:05d}.png".format(test_index))
depth_map = cv2.imread(depth_path, cv2.CV_16U)
depth_map_for_mask = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

pose_path = os.path.join(poses_path, "{:05d}.json".format(test_index))
pose = o3d.io.read_pinhole_camera_parameters(pose_path)

mask = np.copy(depth_map_for_mask) #this mask will be used for OpenCV
mask[np.where(mask > 0)] = 255

# kps_query, descs_query = sift.detectAndCompute(real_img, mask = None)
kps_train, descs_train = sift.detectAndCompute(synth_image, mask = mask)

# temp_matches = matcher.knnMatch(descs_query, descs_train, k=2)

# good_matches = []
# for m, n in temp_matches:
#     if m.distance < ratio_thresh * n.distance:
#         good_matches.append(m)

intrinsics = pose.intrinsic.intrinsic_matrix
fx = intrinsics[0,0]
fy = intrinsics[1,1]
cx = intrinsics[0,2]
cy = intrinsics[1,2]
extrinsics = pose.extrinsic

pointcloud = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_map), pose.intrinsic, extrinsics)
colors = [[0.5, 0.5, 0.6] for i in range(np.asarray(pointcloud.points).shape[0])]
pointcloud.colors = o3d.utility.Vector3dVector(colors)

# for verification
breakpoint()

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.add_geometry(pointcloud)
ctr = vis.get_view_control()
ctr.convert_from_pinhole_camera_parameters(pose, allow_arbitrary=True)
vis.run()
vis.destroy_window()

# debug
# img_matches = np.empty((max(real_img.shape[0], synth_image.shape[0]), real_img.shape[1]+synth_image.shape[1], 3), dtype=np.uint8)
# cv2.drawMatches(real_img, kps_query, synth_image, kps_train, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# #-- Show detected matches
# cv2.imshow('Good Matches', img_matches)
# cv2.waitKey()

print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))