# Remember to get the points average for the model first, get_points_3D_mean_desc_single_model.py
#  and the scale python3 get_scale.py
import os
import subprocess
import sys
import time
import numpy as np
import colmap
from database import COLMAPDatabase
from evaluator import get_Unity_pose_query_image, save_image_projected_points_unity
from feature_matcher_single_image import feature_matcher_wrapper
from point3D_loader import read_points3d_default, get_points3D_xyz_rgb
from pose_solver_single_image import solve, apply_transform_unity

data_dir = sys.argv[1]

def add_ones(x):
    return np.hstack((x,np.ones((x.shape[0], 1))))

K = np.loadtxt("/Users/alex/Projects/CYENS/matrices/pixel_intrinsics_low_640_portrait_unity.txt")

with open("/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/query_name.txt") as f:
    query_frame_name = f.readlines()[0]

db_path = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/database.db"
points3D_path = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/model/0/points3D.bin"
query_images_dir = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/"
image_list_file = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/query_name.txt"
descs_avg_path = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/model/0/avg_descs.npy"
unity_cam_pose = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/cameraPose.txt"
scale_txt_file = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/scale.txt"

subprocess.check_call(["sips", "-r", "-90", os.path.join(query_images_dir, query_frame_name)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

db = COLMAPDatabase.connect(db_path)
points3D = read_points3d_default(points3D_path)
points3D_xyz_rgb = get_points3D_xyz_rgb(points3D) #new model maybe ?

total_elapsed_time = 0
# Step 1: Feature Extractor
print("FE")
start = time.time()
colmap.feature_extractor(db_path, query_images_dir, image_list_file)
end = time.time()
elapsed_time = end - start
total_elapsed_time += elapsed_time
print("Feature Extractor took: " + str(elapsed_time))

# Step 2: Feature Matching
print("FM")
start = time.time()
train_descriptors = np.load(descs_avg_path).astype(np.float32)
matches = feature_matcher_wrapper(db, query_frame_name, train_descriptors, points3D_xyz_rgb, 0.7)
end = time.time()
elapsed_time = end - start
total_elapsed_time += elapsed_time
print("Feature Matching took: " + str(elapsed_time))

# Step 3: Solver
print("S")
start = time.time()
colmap_pose = solve(matches[:,0:5],K)
end = time.time()
elapsed_time = end - start
total_elapsed_time += elapsed_time
print("Solver took: " + str(elapsed_time))

# Step 4: Apply transformation to points
print("AT")
start = time.time()
unity_pose = get_Unity_pose_query_image(unity_cam_pose)

with open(scale_txt_file, 'r') as f:
    scale = f.read()

scale = float(scale)

# points3D_xyz = add_ones(points3D_xyz) # homogeneous
points3DARCore = apply_transform_unity(colmap_pose, unity_pose, scale, points3D_xyz_rgb)
end = time.time()
elapsed_time = end - start
total_elapsed_time += elapsed_time
print("Apply transformations took: " + str(elapsed_time))

print("Debugging")
save_image_projected_points_unity(os.path.join(query_images_dir, query_frame_name), K,
                                  colmap_pose, points3D_xyz_rgb[:,0:4],
                                  os.path.join(query_images_dir, "debug_"+query_frame_name))

print("Saving..")
np.savetxt("/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/points3D_AR.txt", points3DARCore)

print("Total time: " + str(total_elapsed_time))
