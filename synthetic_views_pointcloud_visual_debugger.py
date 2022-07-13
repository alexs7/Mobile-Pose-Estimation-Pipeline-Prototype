import glob
import sys
import time
from os.path import join
import numpy as np
import open3d as o3d
import os
import cv2
from random import *

depth_scale = 1000.0 #Open3D default

start = time.time()

base_path = sys.argv[1] # i.e. /Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/
mesh_path = os.path.join(base_path, "model_files/EXPORT_Mesh/model.fbx")
synth_images_path = os.path.join(base_path, "synth_images/")
images_path = os.path.join(base_path, "IMAGES/all/")
depths_path = os.path.join(base_path, "depths/")
poses_path = os.path.join(base_path, "poses/")
synth_image_features_path = os.path.join(base_path, "images_features/")
no_images = len(glob.glob(os.path.join(synth_images_path, "*.png")))

print("Creating features dir...") # will store 2D - 3D - SIFT in a numpy .npy file for each image
if not os.path.exists(synth_image_features_path):
    os.makedirs(synth_image_features_path)

print("Reading mesh...")
mesh = o3d.io.read_triangle_mesh(mesh_path)

sift = cv2.SIFT_create()

test_index = int(sys.argv[2]) #randint(0, no_images)
print("Random index: " + str(test_index))

synth_image_path = os.path.join(synth_images_path, "{:05d}.png".format(test_index))
synth_image = cv2.imread(synth_image_path, cv2.IMREAD_GRAYSCALE)

depth_path = os.path.join(depths_path, "{:05d}.png".format(test_index))
depth_map = cv2.imread(depth_path, cv2.CV_16U)
depth_map_for_mask = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

pose_path = os.path.join(poses_path, "{:05d}.json".format(test_index))
pose = o3d.io.read_pinhole_camera_parameters(pose_path) # load the pose which is in camera coordinates

mask = np.copy(depth_map_for_mask) #this mask will be used for OpenCV
mask[np.where(mask > 0)] = 255

kps_train, descs_train = sift.detectAndCompute(synth_image, mask = mask)

intrinsics = pose.intrinsic.intrinsic_matrix
fx = intrinsics[0,0]
fy = intrinsics[1,1]
cx = intrinsics[0,2]
cy = intrinsics[1,2]
extrinsics = pose.extrinsic

print("Back-projecting points...")
point_world_coords = np.empty([0,3])
for keypoint in kps_train:
    xy = np.round(keypoint.pt).astype(int) # openCV convention (x,y)
    # numpy convention
    row = xy[1]
    col = xy[0]
    depth = depth_map[row, col]
    z = depth / depth_scale
    x = (xy[0] - cx) * z / fx
    y = (xy[1] - cy) * z / fy

    point_camera_coordinates = np.array([x, y, z , 1]).reshape([4,1])
    point_world_coordinates = np.linalg.inv(extrinsics).dot(point_camera_coordinates)
    point_world_coordinates = point_world_coordinates[0:3,:]
    point_world_coords = np.r_[point_world_coords, point_world_coordinates.reshape([1, 3])]

# TODO: render point-cloud also below for verification
pointcloud_verification = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_map), pose.intrinsic, extrinsics)
colors = [[0, 0.5, 0] for i in range(np.asarray(pointcloud_verification.points).shape[0])]
pointcloud_verification.colors = o3d.utility.Vector3dVector(colors)

pointcloud = o3d.geometry.PointCloud()
pointcloud.points = o3d.utility.Vector3dVector(point_world_coords)
colors = [[0.5, 0.5, 0.6] for i in range(np.asarray(pointcloud.points).shape[0])]
pointcloud.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
vis.create_window()
# vis.add_geometry(mesh)
vis.add_geometry(pointcloud)
vis.add_geometry(pointcloud_verification)


print("Saving Point Cloud..")
vis.capture_depth_point_cloud(os.path.join(base_path, "pointcloud.pcd" ), convert_to_world_coordinate=True)

print("Loading Point Cloud..")
pcd = o3d.io.read_point_cloud("/Users/alex/Projects/CYENS/andreas_models/Model 2 - Old Doorway - Near Green Line/pointcloud.pcd")
colors = [[0.5, 0, 0] for i in range(np.asarray(pcd.points).shape[0])]
pcd.colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(pcd)

ctr = vis.get_view_control()
ctr.convert_from_pinhole_camera_parameters(pose, allow_arbitrary=True)

vis.run()
vis.destroy_window()

print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))