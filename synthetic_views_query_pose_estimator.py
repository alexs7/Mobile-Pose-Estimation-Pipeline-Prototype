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
row_length = 134

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
query_images_path = os.path.join(base_path, "query_images")

if not os.path.exists(verifications_path):
    os.makedirs(verifications_path)

print("Connecting to database...")
db = CYENSDatabase.connect(database_path)

sift = cv2.SIFT_create()
FLANN_INDEX_KDTREE = 1 #https://docs.opencv.org/3.4.0/dc/d8c/namespacecvflann.html
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
matcher = cv2.FlannBasedMatcher(index_params, search_params)
ratio_thresh = 0.7

query_frame = sys.argv[2]
synth_no = int(sys.argv[3]) #150, 12 ...

print("Estimating a pose for image with name: " + query_frame)

query_image_path = os.path.join(query_images_path, query_frame)
query_image = cv2.imread(query_image_path)

synth_image_path = os.path.join(synth_images_path, "{:05d}.png".format(synth_no))
synth_image = cv2.imread(synth_image_path, cv2.IMREAD_GRAYSCALE)

pose_path = os.path.join(poses_path, "{:05d}.json".format(synth_no))
pose = o3d.io.read_pinhole_camera_parameters(pose_path)

kps_query, descs_query = sift.detectAndCompute(query_image, None)
kps_synth, descs_synth = sift.detectAndCompute(synth_image, None)

print("Matching between real and synth image")

matches = matcher.knnMatch(descs_query, descs_synth, k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

good_matches_counter_between_images = 0
for i,(m,n) in enumerate(matches):
    if m.distance < ratio_thresh * n.distance:
        good_matches_counter_between_images += 1
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

synth_query_matches_image = cv2.drawMatchesKnn(query_image, kps_query, synth_image,
                                                kps_synth, matches, None, **draw_params)

synth_query_matches_image_path = os.path.join(verifications_path, "synth_query_matches_image{:06}.png".format(synth_no))
cv2.imwrite(synth_query_matches_image_path, synth_query_matches_image)

print("Matching between real and synth image (db features)")

database_features = db.get_feature_data(str(synth_no), row_length)
descs_train = database_features[:, -128:].astype(np.float32) # last 128 elements (SIFT)

keypoint_image = cv2.drawKeypoints(query_image, kps_query, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
keypoint_image_path = os.path.join(verifications_path, "query_keypoints_frame{:06}.png".format(synth_no))
cv2.imwrite(keypoint_image_path, keypoint_image)

temp_matches = matcher.knnMatch(descs_query, descs_train, k=2)

good_matches = []
for m, n in temp_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

good_query_keypoints = np.array(kps_query)[[good_match.queryIdx for good_match in good_matches]]
keypoints_2D = np.array([good_query_keypoint.pt for good_query_keypoint in good_query_keypoints])
points_3D = database_features[[good_match.trainIdx for good_match in good_matches]][:, 2:5]

print("good_matches_counter_between_images: " + str(good_matches_counter_between_images))
print("good_matches_counter_between_image_and_database: " + str(len(good_matches)))
breakpoint()

K = np.loadtxt(sys.argv[4])

_, rvec, tvec, _ = cv2.solvePnPRansac(points_3D, keypoints_2D, K, np.zeros((5, 1)),
                                      iterationsCount = 3000, confidence = 0.99, flags = cv2.SOLVEPNP_P3P)

rot_matrix = cv2.Rodrigues(rvec)[0] #second value is the jacobian
est_pose_query = np.c_[rot_matrix, tvec]
est_pose_query = np.r_[est_pose_query, [np.array([0, 0, 0, 1])]]

print("Projecting points to verify..")
verification_image_path = os.path.join(verifications_path, "verified_frame{:06}.png".format(synth_no))
save_projected_points(points_3D, keypoints_2D, est_pose_query, pose.intrinsic.intrinsic_matrix, query_image, verification_image_path)

print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))