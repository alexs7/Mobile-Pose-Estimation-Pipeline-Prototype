import glob
import sys
import time
from random import randint

import numpy as np
import open3d as o3d
import os
import cv2
from cyens_database import CYENSDatabase
from save_2D_points import save_projected_points

depth_scale = 1000.0 #Open3D default
data_length = 133

start = time.time()

base_path = sys.argv[1] # i.e. /Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/
mesh_path = os.path.join(base_path, "model_files/EXPORT_Mesh/model.fbx")
synth_images_path = os.path.join(base_path, "synth_images/")
images_path = os.path.join(base_path, "IMAGES/all/")
depths_path = os.path.join(base_path, "depths/")
poses_path = os.path.join(base_path, "poses/")
verifications_path = os.path.join(base_path, "verifications/")
no_images = len(glob.glob(os.path.join(synth_images_path, "*.png")))
database_path = os.path.join(base_path, "features_data.db")

if not os.path.exists(verifications_path):
    os.makedirs(verifications_path)

print("Connecting to database...")
db = CYENSDatabase.connect(database_path)

sift = cv2.SIFT_create()

test_index = randint(0, no_images)

print("Estimating a pose for image with index: " + str(test_index))

real_image_path = os.path.join(images_path, "frame{:06}.png".format(test_index))
real_img = cv2.imread(real_image_path)

synth_image_path = os.path.join(synth_images_path, "{:05d}.png".format(test_index))
synth_image = cv2.imread(synth_image_path, cv2.IMREAD_GRAYSCALE)

pose_path = os.path.join(poses_path, "{:05d}.json".format(test_index))
pose = o3d.io.read_pinhole_camera_parameters(pose_path)

kps_query, descs_query = sift.detectAndCompute(real_img, None)
database_features = db.get_feature_data(test_index, data_length)
descs_train = database_features[:,5:data_length].astype(np.float32)

FLANN_INDEX_KDTREE = 1 #https://docs.opencv.org/3.4.0/dc/d8c/namespacecvflann.html
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
matcher = cv2.FlannBasedMatcher(index_params, search_params)
ratio_thresh = 0.7

temp_matches = matcher.knnMatch(descs_query, descs_train, k=2)

good_matches = []
for m, n in temp_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

good_query_keypoints = np.array(kps_query)[[good_match.queryIdx for good_match in good_matches]]
keypoints_2D = np.array([good_query_keypoint.pt for good_query_keypoint in good_query_keypoints])
points_3D = database_features[[good_match.trainIdx for good_match in good_matches]][:, 2:5]

_, rvec, tvec, _ = cv2.solvePnPRansac(points_3D, keypoints_2D, pose.intrinsic.intrinsic_matrix, np.zeros((5, 1)),
                                      iterationsCount = 3000, confidence = 0.99, flags = cv2.SOLVEPNP_P3P)

rot_matrix = cv2.Rodrigues(rvec)[0] #second value is the jacobian
est_pose_query = np.c_[rot_matrix, tvec]
est_pose_query = np.r_[est_pose_query, [np.array([0, 0, 0, 1])]]

print("Projecting points ")

print("Projecting points to verify..")
verification_image_path = os.path.join(verifications_path, "verified_frame{:06}.png".format(test_index))
save_projected_points(points_3D, keypoints_2D, est_pose_query, pose.intrinsic.intrinsic_matrix, real_img, verification_image_path)

print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))