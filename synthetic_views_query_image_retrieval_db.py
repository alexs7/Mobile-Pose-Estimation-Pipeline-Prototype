# This file will be used to extract local descs from query images ang global ones.
# Then will estimate a pose
# TODO: https://stackoverflow.com/questions/52865771/write-opencv-image-in-memory-to-bytesio-or-tempfile - to load images from phone
import os
from pathlib import Path
import h5py
import pycolmap
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, localize_sfm
from hloc.utils.io import get_matches, get_keypoints, find_pair, read_image
from hloc.utils.parsers import parse_retrieval
from hloc.utils.viz import plot_images, plot_matches, save_plot
from tqdm import tqdm
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import cv2
import re
from synthetic_views_pose_solvers import opencv_pose, pycolmap_wrapper_absolute_pose, project_points

from save_2D_points import save_points_2D_2D

query_images = Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query_images/')
db_descriptors = '/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/db/global-feats-openibl.h5'
mesh_data_path = '/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/mesh_data/'
db_local_descriptors = '/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/db/feats-superpoint-n4096-rmax1600.h5'
query_local_descriptors = '/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query/feats-superpoint-n4096-rmax1600.h5'
query_outputs = Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query')
query_image_retrieval_pairs = query_outputs / 'pairs.txt'
K_nearest_images = 1 #TODO: Double check what happens when you increase K - pose error seems to be worse instead of better ? (maybe need to remove duplicates in points 2D list)
unity_K_matrix_portrait = np.loadtxt('/home/alex/AR_CYENS/fullpipeline/matrices/pixel_intrinsics_low_640_portrait_unity.txt')
width = 480
height = 640
fx = unity_K_matrix_portrait[0,0]
fy = unity_K_matrix_portrait[1,1]
cx = unity_K_matrix_portrait[0,2]
cy = unity_K_matrix_portrait[1,2]
query_image_name = 'frame_25.jpg'
q_image = cv2.imread(os.path.join('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query_images', query_image_name))

camera = pycolmap.Camera(
    model='PINHOLE',
    width=width,
    height=height,
    params=[fx, fy, cx, cy],
)

retrieval_conf = extract_features.confs['openibl'] #'dir' = ResNet - deep image retrieval
feature_conf = extract_features.confs['superpoint_max']
matcher_conf = match_features.confs['superglue']

# get the K closest db images for each query image
query_descriptors = extract_features.main(retrieval_conf, query_images, query_outputs)
pairs_from_retrieval.main(query_descriptors, query_image_retrieval_pairs, num_matched=K_nearest_images, db_descriptors = Path(db_descriptors))

# match the query images to their already retrieved closest neighbours (K=3 or whatever)
query_features = extract_features.main(feature_conf, query_images, query_outputs)
matches = match_features.main(matcher_conf, query_image_retrieval_pairs, feature_conf['output'], query_outputs, features_ref = Path(db_local_descriptors))
ret = parse_retrieval(query_image_retrieval_pairs)

points_2D_all = np.empty([0,2])
points_3D_all = np.empty([0,3])
for i in range(K_nearest_images):

    # sample query frame - TODO: replace with actual query name (or pass opencv image in bytes ?)
    db_image_name = ret['frame_25.jpg'][i]
    db_image_number = re.findall(r"\d+", db_image_name)[0]
    points_3D_db = np.load(os.path.join(mesh_data_path, "points_3D_" + db_image_number + ".npy")) # TODO: change this to a database query

    pair = (query_image_name, db_image_name)
    matches_idxs , _ = get_matches(matches, query_image_name, db_image_name)

    kp_q_xy = get_keypoints(Path(query_local_descriptors), query_image_name)
    kp_db_xy = get_keypoints(Path(db_local_descriptors), db_image_name)
    kp_db_xy = np.round(kp_db_xy[matches_idxs[:, 1]]).astype(np.int16)

    points_2D = np.round(kp_q_xy[matches_idxs[:, 0]])
    points_3D = points_3D_db[kp_db_xy[:, 1], kp_db_xy[:, 0]]  # note the reverse here

    points_2D_all = np.r_[points_2D_all, points_2D]
    # points_3D are in world space given depth is consistent across frames (double check, so maybe reconsider saving the open3D poses in world space ?)
    points_3D_all = np.r_[points_3D_all, points_3D]

    gt_pose = o3d.io.read_pinhole_camera_parameters(os.path.join(mesh_data_path,"pose_in_cc_" + db_image_number + ".json"))  # load the pose which is in camera coordinates (?)

    db_est_points_2D = project_points(points_3D, gt_pose.intrinsic.intrinsic_matrix, np.linalg.inv(gt_pose.extrinsic))
    db_image = cv2.imread(os.path.join(mesh_data_path, db_image_name)).copy()
    save_points_2D_2D(db_est_points_2D, kp_db_xy, db_image,
                      os.path.join("/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/vis_debug/", db_image_name), est_size=10, gt_size=8)

    breakpoint()

opencv_pose = opencv_pose(points_2D_all, points_3D_all, unity_K_matrix_portrait)
pycolmap_pose = pycolmap_wrapper_absolute_pose(points_2D_all, points_3D_all, camera)

opencv_2D_points = project_points(points_3D_all, unity_K_matrix_portrait, opencv_pose)
pycolmap_2D_points = project_points(points_3D_all, unity_K_matrix_portrait, pycolmap_pose)

save_points_2D_2D(opencv_2D_points, points_2D_all, q_image, "/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/vis_debug/result_opencv.png")
save_points_2D_2D(pycolmap_2D_points, points_2D_all, q_image, "/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/vis_debug/result_pycolmap.png")

breakpoint()

# debug vis matches
# uncomment to save visuals for matches for all pairs
# ret = parse_retrieval(query_image_retrieval_pairs)
# pairs = [(q, r) for q, rs in ret.items() for r in rs]
# for pair in tqdm(pairs):
#     matches_kp_idx_img_1_idx_2 , _ = get_matches(matches, pair[0], pair[1])
#     kp_q = get_keypoints(Path(query_local_descriptors), pair[0])
#     kp_db = get_keypoints(Path(db_local_descriptors), pair[1])
#
#     q_image = read_image(Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/hloc/query_images') / pair[0])
#     db_image = read_image(Path('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/mesh_data/') / pair[1])
#     plot_images([q_image, db_image], dpi=100)
#     plot_matches(kp_q[matches_kp_idx_img_1_idx_2[:,0]], kp_db[matches_kp_idx_img_1_idx_2[:,1]], lw=2.5, color=(0,1,0), a=.1)
#     save_plot('/media/iNicosiaData/data/andreas_models/Model 6 - Plateia Dimarchon - Fragment/vis_debug/' + pair[0]+"_"+pair[1])