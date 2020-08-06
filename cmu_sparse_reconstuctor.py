import sys
import os
import colmap
from arrange_sessions import gen_query_txt, gen_base_cam_centers_txt
import subprocess
import glob
from get_scale import calc_scale_COLMAP, calc_scale_COLMAP_ARCORE
import sys
import numpy as np

# In order for this to work you have to transfer the images manually into the correct folders first
# Remember to undistort images first
path = sys.argv[1] # i.e /home/alex/fullpipeline/colmap_data/CMU_data/slice4/
slice = path.split('/')[-2]
arcore = sys.argv[2] == '1'

#base mode paths
base_db_path = path+"base/database.db"
base_images_dir = path+"base/images"
base_model_dir = path+"base/model"
reference_model_images_path = "/home/alex/Datasets/Extended-CMU-Seasons/Extended-CMU-Seasons/"+slice+"/ground-truth-database-images-"+slice+".txt"
alignment_reference_cam_centers_txt = path+"base/base_images_cam_centers.txt"

base_images_no = len(glob.glob1(base_images_dir,"*.jpg"))

colmap.feature_extractor(base_db_path, base_images_dir)
colmap.vocab_tree_matcher(base_db_path)
colmap.mapper(base_db_path, base_images_dir, base_model_dir)

if(arcore):
    scale = calc_scale_COLMAP_ARCORE("/home/alex/Mobile-Pose-Estimation-Pipeline-Prototype/colmap_data/local_datasets/Coop/reference_data/base_reference_data/", base_model_dir+"/0/images.bin")
    np.savetxt(path+"scale.txt", [scale])
else:
    gen_base_cam_centers_txt(base_images_dir, reference_model_images_path)
    # Note: this will overwrite the first model
    colmap.model_aligner(base_model_dir+"/0", base_model_dir+"/0", alignment_reference_cam_centers_txt)

query_images_dir = path+"live/images/"
gen_query_txt(query_images_dir, base_images_no)

# live mode paths
live_db_path = path+"live/database.db"
live_images_dir = path+"live/images"
live_model_dir = path+"live/model"
live_query_image_list_file = path+"live/query_name.txt"

subprocess.run(["cp", base_db_path, live_db_path])

colmap.feature_extractor(live_db_path, live_images_dir, live_query_image_list_file, query=True)
colmap.vocab_tree_matcher(live_db_path, live_query_image_list_file)
colmap.image_registrator(live_db_path, base_model_dir+"/0", live_model_dir)

query_images_dir = path+"gt/images/"
gen_query_txt(query_images_dir)

# gt mode paths
gt_db_path = path+"gt/database.db"
gt_images_dir = path+"gt/images"
gt_model_dir = path+"gt/model"
gt_query_image_list_file = path+"gt/query_name.txt"

subprocess.run(["cp", live_db_path, gt_db_path])

colmap.feature_extractor(gt_db_path, gt_images_dir, gt_query_image_list_file, query=True)
colmap.vocab_tree_matcher(gt_db_path, gt_query_image_list_file)
colmap.image_registrator(gt_db_path, live_model_dir, gt_model_dir)
