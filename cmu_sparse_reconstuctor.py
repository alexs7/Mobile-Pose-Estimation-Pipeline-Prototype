import sys
import os
import colmap
from arrange_sessions import gen_query_txt
import subprocess
import glob
from get_scale import calc_scale_COLMAP
import sys
import numpy as np

# In order for this to work you have to transfer the images manually into the correct folders first
original_images_path = sys.argv[1]

#base mode paths
base_db_path = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/base/database.db"
base_images_dir = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/base/images"
base_model_dir = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/base/model"

base_images_no = len(glob.glob1(base_images_dir,"*.jpg"))

colmap.feature_extractor(base_db_path, base_images_dir)
colmap.vocab_tree_matcher(base_db_path)
colmap.mapper(base_db_path, base_images_dir, base_model_dir)

scale = calc_scale_COLMAP(original_images_path, base_model_dir+"/0/images.bin")
np.savetxt("/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/scale.txt", [scale])

query_images_dir = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/live/images/"
gen_query_txt(query_images_dir, base_images_no)

# live mode paths
live_db_path = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/live/database.db"
live_images_dir = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/live/images"
live_model_dir = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/live/model"
live_query_image_list_file = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/live/query_name.txt"

subprocess.run(["cp", base_db_path, live_db_path])

colmap.feature_extractor(live_db_path, live_images_dir, live_query_image_list_file, query=True)
colmap.vocab_tree_matcher(live_db_path, live_query_image_list_file)
colmap.image_registrator(live_db_path, base_model_dir+"/0", live_model_dir)

query_images_dir = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/gt/images/"
gen_query_txt(query_images_dir)

# gt mode paths
gt_db_path = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/gt/database.db"
gt_images_dir = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/gt/images"
gt_model_dir = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/gt/model"
gt_query_image_list_file = "/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/data/gt/query_name.txt"

subprocess.run(["cp", live_db_path, gt_db_path])

colmap.feature_extractor(gt_db_path, gt_images_dir, gt_query_image_list_file, query=True)
colmap.vocab_tree_matcher(gt_db_path, gt_query_image_list_file)
colmap.image_registrator(gt_db_path, live_model_dir, gt_model_dir)
