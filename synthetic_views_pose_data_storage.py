import glob
import sys
import time
import numpy as np
import open3d as o3d
import os
import cv2
from cyens_database import CYENSDatabase

depth_scale = 1000.0 #Open3D default
data_length = 133

start = time.time()

base_path = sys.argv[1] # i.e. /Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/
mesh_path = os.path.join(base_path, "model_files/EXPORT_Mesh/model.fbx")
synth_images_path = os.path.join(base_path, "synth_images/")
images_path = os.path.join(base_path, "IMAGES/all/")
depths_path = os.path.join(base_path, "depths/")
poses_path = os.path.join(base_path, "poses/")
no_images = len(glob.glob(os.path.join(synth_images_path, "*.png")))
database_path = os.path.join(base_path, "features_data.db")

print("Creating database...")
db = CYENSDatabase.connect(database_path)
db.create_data_table()

sift = cv2.SIFT_create()

for i in range(no_images):

    print("Doing image with index: " + str(i))

    synth_image_path = os.path.join(synth_images_path, "{:05d}.png".format(i))
    synth_image = cv2.imread(synth_image_path, cv2.IMREAD_GRAYSCALE)

    depth_path = os.path.join(depths_path, "{:05d}.png".format(i))
    depth_map = cv2.imread(depth_path, cv2.CV_16U)
    depth_map_for_mask = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    pose_path = os.path.join(poses_path, "{:05d}.json".format(i))
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

    data_rows = np.empty([0, data_length])
    for k in range(len(kps_train)):
        keypoint = kps_train[k]
        descriptor = descs_train[k] #same order as above
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
        data_row = np.append(np.array([xy[0], xy[1], x, y, z]), descriptor).reshape([1, data_length])
        data_rows = np.r_[data_rows, data_row]

    db.add_feature_data(i, data_rows)

print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))