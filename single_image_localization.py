# Remember to get the points average for the model first, get_points_3D_mean_desc_single_model.py
#  and the scale python3 get_scale.py

import sys
import time
import numpy as np
import colmap
from database import COLMAPDatabase
from evaluator import get_Unity_pose_query_image, save_image_projected_points_unity
from feature_matcher_single_image import feature_matcher_wrapper
from point3D_loader import read_points3d_default, get_points3D_xyz_rgb
from pose_solver_single_image import solve, apply_transform_unity

def add_ones(x):
    return np.hstack((x,np.ones((x.shape[0], 1))))

K = np.loadtxt("/Users/alex/Projects/CYENS/matrices/pixel_intrinsics_low_640_landscape.txt")
query_frame_name = sys.argv[1] # same name as in query_name.txt but just the filename
with open('/Users/alex/Projects/CYENS/unity_electron_query_images/query_name.txt', "w") as myfile:
    myfile.write(query_frame_name)
db_path = "/Users/alex/Projects/CYENS/colmap_models/database.db"
points3D_path = "/Users/alex/Projects/CYENS/colmap_models/model/0/points3D.bin"
query_images_dir = "/Users/alex/Projects/CYENS/unity_electron_query_images"
image_list_file = "/Users/alex/Projects/CYENS/unity_electron_query_images/query_name.txt"
descs_avg_path = "/Users/alex/Projects/CYENS/colmap_models/model/0/avg_descs.npy"
unity_cam_pose = "/Users/alex/Projects/CYENS/unity_electron_query_images/cameraPose.txt"

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
scale =  0.09132078648683176
# points3D_xyz = add_ones(points3D_xyz) # homogeneous
points3DARCore = apply_transform_unity(colmap_pose, unity_pose, scale, points3D_xyz_rgb)
end = time.time()
elapsed_time = end - start
total_elapsed_time += elapsed_time
print("Apply transformations took: " + str(elapsed_time))

print("Debugging")
#make this generic
save_image_projected_points_unity("/Users/alex/Projects/CYENS/unity_electron_query_images/frame_8.jpg", K, colmap_pose, points3D_xyz_rgb[:,0:4])

print("Saving..")
np.savetxt('/Users/alex/Projects/CYENS/colmap_models/points3D_AR.txt', points3DARCore)

print("Total time: " + str(total_elapsed_time))
