import glob
import sys
import time
from random import randint
from  sklearn.metrics import mean_absolute_error
from  sklearn.metrics import mean_squared_error
import numpy as np
import open3d as o3d
import os
import cv2
from cyens_database import CYENSDatabase
from save_2D_points import save_projected_points, save_points_2D_2D

np.set_printoptions(precision=3, suppress=True)

depth_scale = 1000.0 #Open3D default
row_length = 134

base_path = sys.argv[1] # i.e. /Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/
synth_images_path = os.path.join(base_path, "synth_images/")
# images_path = os.path.join(base_path, "IMAGES/all/")
depths_path = os.path.join(base_path, "depths/")
verifications_path = os.path.join(base_path, "query_verifications/")
no_images = len(glob.glob(os.path.join(synth_images_path, "*.png")))
database_path = os.path.join(base_path, "features_data_ccs.db")
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

query_image_path = os.path.join(query_images_path, query_frame)
query_image = cv2.imread(query_image_path)

synth_image_path = os.path.join(synth_images_path, "{:05d}.png".format(synth_no))
synth_image = cv2.imread(synth_image_path)

kps_query, descs_query = sift.detectAndCompute(query_image, None)
kps_query_xy = np.round(cv2.KeyPoint_convert(kps_query)).astype(int)

# mask
depth_path = os.path.join(depths_path, "{:05d}.png".format(synth_no))
depth_float_path = os.path.join(depths_path, "{:05d}_float.npy".format(synth_no))
depth_map_for_mask = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
mask = np.copy(depth_map_for_mask)  # this mask will be used for OpenCV
mask[np.where(mask > 0)] = 255

kps_synth, descs_synth = sift.detectAndCompute(synth_image, mask=mask)

# 1/2 match to create validation image
matches = matcher.knnMatch(descs_query, descs_synth, k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

good_matches = []
for i,(m,n) in enumerate(matches):
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

synth_query_matches_image = cv2.drawMatchesKnn(query_image, kps_query, synth_image,
                                                kps_synth, matches, None, **draw_params)

synth_query_matches_image_path = os.path.join(verifications_path, "synth_query_matches_image_{:06}.png".format(synth_no))
cv2.imwrite(synth_query_matches_image_path, synth_query_matches_image)

# 2/2 match again to get the pose
start = time.time()

database_features = db.get_feature_data(str(synth_no), row_length)
matches = matcher.knnMatch(descs_query, database_features[:, -128:].astype(np.float32), k=2)

good_matches = []
for i,(m,n) in enumerate(matches):
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

points_2D_idx = np.array([good_match.queryIdx for good_match in good_matches])
points_2D = kps_query_xy[points_2D_idx,:].astype(np.float32)

points_3D_idx = np.array([good_match.trainIdx for good_match in good_matches])
points_3D = database_features[points_3D_idx, 2:5].astype(np.float32)

K = np.eye(4)
K[0:3, 0:3] = np.loadtxt(sys.argv[4])

_, rvec, tvec, _ = cv2.solvePnPRansac(points_3D, points_2D, K[0:3,0:3], distCoeffs = np.zeros((5, 1)), iterationsCount = 10000, confidence = 0.99, flags = cv2.SOLVEPNP_EPNP)

rot_matrix = cv2.Rodrigues(rvec)[0] #second value is the jacobian
est_pose_query = np.c_[rot_matrix, tvec]
est_pose_query = np.r_[est_pose_query, [np.array([0, 0, 0, 1])]]

print("Projecting points to verify..")
points_3D = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))
points_2D_est = K.dot(est_pose_query.dot(points_3D.transpose()))
points_2D_est = points_2D_est / points_2D_est[2, :]
# Note that some points will not show up because they are outliers.
# To visually check this: look at the matches from the query image to the
# synth image. Some query points are matched to wrong synth points.
# RANSAC sees them as outliers and discards them, that is why some no blue points over green
# This should return [u, v , 1 , 1/z], https://medium.com/yodayoda/from-depth-map-to-point-cloud-7473721d3f
points_2D_est = points_2D_est.transpose()[:, 0:2]

verification_image_path = os.path.join(verifications_path, query_frame)
save_points_2D_2D(points_2D_est, points_2D, query_image, verification_image_path)

print("Keypoints no: " + str(points_2D.shape[0]))
mae = mean_absolute_error(points_2D_est, points_2D)
print("MAE: " + str(mae))
rmse = mean_squared_error(points_2D_est, points_2D, squared=False)
print("RMSE: " + str(rmse))

print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))

# Old code:

# print("Estimating a pose for image with name: " + query_frame)
#
# query_image_path = os.path.join(query_images_path, query_frame)
# query_image = cv2.imread(query_image_path)
#
# synth_image_path = os.path.join(synth_images_path, "original_{:05d}.png".format(synth_no))
# synth_image_color = cv2.imread(synth_image_path)
# synth_image = cv2.imread(synth_image_path, cv2.IMREAD_GRAYSCALE)
#
# kps_query, descs_query = sift.detectAndCompute(query_image, None)
# kps_synth, descs_synth = sift.detectAndCompute(synth_image, None)
#
# keypoint_image = cv2.drawKeypoints(query_image, kps_query, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
# keypoint_image_path = os.path.join(verifications_path, "query_keypoints_frame_{:06}.png".format(synth_no))
# cv2.imwrite(keypoint_image_path, keypoint_image)
#
# print("Matching between real and synth image db descs")
#
# matches = matcher.knnMatch(descs_query, descs_synth, k=2)
# # Need to draw only good matches, so create a mask
# matchesMask = [[0,0] for i in range(len(matches))]
#
# good_matches_counter_between_images = 0
# for i,(m,n) in enumerate(matches):
#     if m.distance < ratio_thresh * n.distance:
#         good_matches_counter_between_images += 1
#         matchesMask[i]=[1,0]
#
# draw_params = dict(matchColor = (0,255,0),
#                    singlePointColor = (255,0,0),
#                    matchesMask = matchesMask,
#                    flags = cv2.DrawMatchesFlags_DEFAULT)
#
# synth_query_matches_image = cv2.drawMatchesKnn(query_image, kps_query, synth_image,
#                                                 kps_synth, matches, None, **draw_params)
#
# synth_query_matches_image_path = os.path.join(verifications_path, "synth_query_matches_image_{:06}.png".format(synth_no))
# cv2.imwrite(synth_query_matches_image_path, synth_query_matches_image)
#
# print("Matching between real and synth image (db features)..")
#
# database_features = db.get_feature_data(str(synth_no), row_length)
# descs_train = database_features[:, -128:].astype(np.float32) # last 128 elements (SIFT)
#
# temp_matches = matcher.knnMatch(descs_query, descs_train, k=2)
#
# good_matches = []
# for m, n in temp_matches:
#     if m.distance < ratio_thresh * n.distance:
#         good_matches.append(m)
#
# # get good query indices
# good_query_keypoints = np.array(kps_query)[[good_match.queryIdx for good_match in good_matches]]
# # get good keypoints 2D
# keypoints_2D = np.array([good_query_keypoint.pt for good_query_keypoint in good_query_keypoints])
# # get good 3D points
# points_3D = database_features[[good_match.trainIdx for good_match in good_matches]][:, 2:5]
# # get good 2D points
# db_image_2D = database_features[[good_match.trainIdx for good_match in good_matches]][:, 0:2]
#
# for point in db_image_2D:
#     cv2.circle(synth_image_color, (int(point[0]), int(point[1])), 4, (0,255,0), -1)
# synth_image_db_keypoints_path = os.path.join(verifications_path, "synth_image_db_keypoints_path_{:06}.png".format(synth_no))
# cv2.imwrite(synth_image_db_keypoints_path, synth_image_color)
#
# print("good_matches_counter_between_images: " + str(good_matches_counter_between_images))
# print("good_matches_counter_between_image_and_database: " + str(len(good_matches)))
#
# K = np.loadtxt(sys.argv[4])
#
# _, rvec, tvec, _ = cv2.solvePnPRansac(points_3D, keypoints_2D, K, np.zeros((5, 1)),
#                                       iterationsCount = 3000, confidence = 0.99, flags = cv2.SOLVEPNP_P3P)
#
# rot_matrix = cv2.Rodrigues(rvec)[0] #second value is the jacobian
# est_pose_query = np.c_[rot_matrix, tvec]
# est_pose_query = np.r_[est_pose_query, [np.array([0, 0, 0, 1])]]
#
# print("Projecting points to verify..")
# verification_image_path = os.path.join(verifications_path, "verified_frame{:06}.png".format(synth_no))
# save_projected_points(points_3D, keypoints_2D, est_pose_query, K, query_image, verification_image_path)
#
# print("Done!...")
# end = time.time()
# elapsed_time = end - start
# print("Time taken (s): " + str(elapsed_time))