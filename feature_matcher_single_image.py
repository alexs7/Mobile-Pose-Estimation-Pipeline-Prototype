from itertools import chain
import cv2
import numpy as np

# creates 2d-3d matches data for ransac comparison
def get_keypoints_xy(db, image_id):
    query_image_keypoints_data = db.execute("SELECT data FROM keypoints WHERE image_id = " + "'" + image_id + "'")
    query_image_keypoints_data = query_image_keypoints_data.fetchone()[0]
    query_image_keypoints_data_cols = db.execute("SELECT cols FROM keypoints WHERE image_id = " + "'" + image_id + "'")
    query_image_keypoints_data_cols = int(query_image_keypoints_data_cols.fetchone()[0])
    query_image_keypoints_data = db.blob_to_array(query_image_keypoints_data, np.float32)
    query_image_keypoints_data_rows = int(np.shape(query_image_keypoints_data)[0] / query_image_keypoints_data_cols)
    query_image_keypoints_data = query_image_keypoints_data.reshape(query_image_keypoints_data_rows, query_image_keypoints_data_cols)
    query_image_keypoints_data_xy = query_image_keypoints_data[:, 0:2]
    return query_image_keypoints_data_xy

# indexing is the same as points3D indexing for trainDescriptors
def get_queryDescriptors(db, image_id):
    query_image_descriptors_data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + image_id + "'")
    query_image_descriptors_data = query_image_descriptors_data.fetchone()[0]
    query_image_descriptors_data = db.blob_to_array(query_image_descriptors_data, np.uint8)
    descs_rows = int(np.shape(query_image_descriptors_data)[0] / 128)
    query_image_descriptors_data = query_image_descriptors_data.reshape([descs_rows, 128])

    row_sums = query_image_descriptors_data.sum(axis=1)
    query_image_descriptors_data = query_image_descriptors_data / row_sums[:, np.newaxis]
    queryDescriptors = query_image_descriptors_data.astype(np.float32)
    return queryDescriptors

def get_image_id(db, query_image):
    image_id = db.execute("SELECT image_id FROM images WHERE name = " + "'" + query_image + "'")
    image_id = str(image_id.fetchone()[0])
    return image_id

def feature_matcher_wrapper(db, query_image, trainDescriptors, points3D_xyz_rgb, ratio_test_val, points_scores_array=None):
    image_id = get_image_id(db,query_image)
    # keypoints data
    keypoints_xy = get_keypoints_xy(db, image_id)
    queryDescriptors = get_queryDescriptors(db, image_id)

    #FLANN
    FLANN_INDEX_KDTREE = 1 #https://docs.opencv.org/3.4.0/dc/d8c/namespacecvflann.html
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    #or BruteForce
    #matcher = cv2.BFMatcher()  # cv2.FlannBasedMatcher(Parameters.index_params, Parameters.search_params) # or cv.BFMatcher()

    # Matching on trainDescriptors (remember these are the means of the 3D points)
    temp_matches = matcher.knnMatch(queryDescriptors, trainDescriptors, k=2)

    # output: idx1, idx2, lowes_distance (vectors of corresponding indexes in
    # m the closest, n is the second closest
    good_matches = np.empty([0, 9])
    for m, n in temp_matches: # TODO: maybe consider what you have at this point? and add it to the if condition ?
        assert(m.distance <= n.distance)
        # trainIdx is from 0 to no of points 3D (since each point 3D has a desc), so you can use it as an index here
        if (m.distance < ratio_test_val * n.distance): #and (score_m > score_n):
            if(m.queryIdx >= keypoints_xy.shape[0]): #keypoints_xy.shape[0] always same as queryDescriptors.shape[0]
                raise Exception("m.queryIdx error!")
            if (m.trainIdx >= points3D_xyz_rgb.shape[0]):
                raise Exception("m.trainIdx error!")
            # idx1.append(m.queryIdx)
            # idx2.append(m.trainIdx)
            scores = []
            xy2D = keypoints_xy[m.queryIdx, :].tolist()
            xyz3D_rgb = points3D_xyz_rgb[m.trainIdx, :].tolist()

            if (points_scores_array is not None):
                for points_scores in points_scores_array:
                    scores.append(points_scores[0, m.trainIdx])
                    scores.append(points_scores[0, n.trainIdx])

            match_data = [xy2D, xyz3D_rgb]
            match_data = list(chain(*match_data))
            good_matches = np.r_[good_matches, np.reshape(match_data, [1,9])]

    return good_matches
