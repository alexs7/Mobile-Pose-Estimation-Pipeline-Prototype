# Remember to get the points average for the model first,
#  1) get_points_3D_mean_desc_single_model.py
#  2) and the scale python3 get_scale.py

# Sample output (with times and a model of 25k points)
# Setting up..
# Setting up took: 2.5194039344787598
# Feature Extractor took: 1.2238779067993164 (COLMAP reports: 0.014 [minutes])
# Feature Matching took: 1.1360161304473877
# Solver took: 0.030557870864868164
# Apply transformations took: 0.00560307502746582
# Total time: 4.915458917617798

import sys
import time
import numpy as np
import colmap
from database import COLMAPDatabase
from evaluator import get_ARCore_pose_query_image
from feature_matcher_single_image import feature_matcher_wrapper
from point3D_loader import read_points3d_default, get_points3D_xyz
from pose_solver_single_image import solve, apply_transform

def add_ones(x):
    return np.hstack((x,np.ones((x.shape[0], 1))))

K = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_landscape.txt")
query_frame_name = sys.argv[1] # same name as in query_name.txt but just the filename
db_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/database.db" #this doesn't need to be the original one it can be diff
points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/model/0/points3D.bin"
query_images_dir = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/current_query_image"
image_list_file = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_name.txt" #the names here are relative to the path "query_images_dir/"
descs_avg_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/descriptors_avg/avg_descs.npy"

total_elapsed_time = 0
start = time.time()
print("Setting up..")
db = COLMAPDatabase.connect(db_path)
points3D = read_points3d_default(points3D_path)
points3D_xyz = get_points3D_xyz(points3D) #new model maybe ?
end = time.time()
elapsed_time = end - start
total_elapsed_time += elapsed_time
print("Setting up took: " + str(elapsed_time))

# Step 1: Feature Extractor
start = time.time()
colmap.feature_extractor(db_path, query_images_dir, image_list_file)
end = time.time()
elapsed_time = end - start
total_elapsed_time += elapsed_time
print("Feature Extractor took: " + str(elapsed_time))

# Step 2: Feature Matching
start = time.time()
train_descriptors = np.load(descs_avg_path).astype(np.float32)
matches = feature_matcher_wrapper(db, query_frame_name, train_descriptors, points3D_xyz, 0.65)
end = time.time()
elapsed_time = end - start
total_elapsed_time += elapsed_time
print("Feature Matching took: " + str(elapsed_time))

# Step 3: Solver
start = time.time()
colmap_pose = solve(matches,K)
end = time.time()
elapsed_time = end - start
total_elapsed_time += elapsed_time
print("Solver took: " + str(elapsed_time))

# Step 4: Apply transformation to points
start = time.time()
arcore_pose = get_ARCore_pose_query_image()
scale = 0.39034945701174184 #get_scale.py
points3D_xyz = add_ones(points3D_xyz) # homogeneous
points3DARCore = apply_transform(colmap_pose, arcore_pose, scale, points3D_xyz)
end = time.time()
elapsed_time = end - start
total_elapsed_time += elapsed_time
print("Apply transformations took: " + str(elapsed_time))

np.savetxt('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt', points3DARCore)

print("Total time: " + str(total_elapsed_time))
