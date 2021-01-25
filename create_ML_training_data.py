import cv2
import numpy as np
from tensorflow import keras

import colmap
from database import COLMAPDatabase
from feature_matcher_single_image import get_image_id, get_keypoints_xy, get_queryDescriptors

db_path = "/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/all_data_and_models/coop_local/ML_training_data/database.db"
query_images_dir = "/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/all_data_and_models/coop_local/ML_training_data/only_jpgs"
image_list_file = "/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/all_data_and_models/coop_local/ML_training_data/query_name.txt"

colmap.feature_extractor(db_path, query_images_dir, image_list_file)

db = COLMAPDatabase.connect(db_path)

with open(image_list_file) as f:
    query_images = f.readlines()
query_images = [x.strip() for x in query_images]

pointcloud3D_xyz_score_sift = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/coop_local/points3D_sorted_descending_heatmap_per_image.txt")
trainDescriptors = pointcloud3D_xyz_score_sift[:,4:132].astype(np.float32)

print("Matching..")
matcher = cv2.BFMatcher()
ratio_test_val = 0.65
training_data_matches = np.empty([0, 129])

print()
for i in range(len(query_images)):
    print("Matching image " + str(i + 1) + "/" + str(len(query_images)), end="\r")
    q_img = query_images[i]
    image_id = get_image_id(db, q_img)
    # keypoints data
    keypoints_xy = get_keypoints_xy(db, image_id)
    queryDescriptors = get_queryDescriptors(db, image_id)

    temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

    for m, n in temp_matches:
        assert (m.distance <= n.distance)
        if (m.distance < ratio_test_val * n.distance): #passes ratio test
            if (m.queryIdx >= keypoints_xy.shape[0]):  # keypoints_xy.shape[0] always same as queryDescriptors.shape[0]
                raise Exception("m.queryIdx error!")
            if (m.trainIdx >= trainDescriptors.shape[0]):
                raise Exception("m.trainIdx error!")

            queryDescriptor = queryDescriptors[m.queryIdx, :]
            pointcloud_score = pointcloud3D_xyz_score_sift[m.trainIdx, 3]

            match_data = np.append(queryDescriptor, pointcloud_score)
            training_data_matches = np.r_[training_data_matches, np.reshape(match_data, [1, 129])]
        else: #doesnt pass ratio test
            if (m.queryIdx >= keypoints_xy.shape[0]):  # keypoints_xy.shape[0] always same as queryDescriptors.shape[0]
                raise Exception("m.queryIdx error!")
            if (m.trainIdx >= trainDescriptors.shape[0]):
                raise Exception("m.trainIdx error!")

            queryDescriptor = queryDescriptors[m.queryIdx, :]
            pointcloud_score = 0

            match_data = np.append(queryDescriptor, pointcloud_score)
            training_data_matches = np.r_[training_data_matches, np.reshape(match_data, [1, 129])]

print()
breakpoint()