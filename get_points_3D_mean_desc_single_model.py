# This is used for single frame localization from a phone:
# Example: python3 get_points_3D_mean_desc_single_model.py /home/alex/AR_CYENS/data/bedroom3/

import os
import sys

import numpy as np
from database import COLMAPDatabase
from point3D_loader import read_points3d_default, index_dict
from query_image import read_images_binary, get_images_ids, get_images_names_from_sessions_numbers

def get_desc_avg(points3D, db):
    points_mean_descs = np.empty([0, 128])

    for k,v in points3D.items():
        point_id = v.id
        points3D_descs = np.empty([0, 128])
        points_image_ids = points3D[point_id].image_ids #COLMAP adds the image twice some times.
        # Loop through the points' image ids and check if it is seen by any image_ids
        # If it is seen then get the desc for each id.
        for k in range(len(points_image_ids)):
            id = points_image_ids[k]
            data = db.execute("SELECT data FROM descriptors WHERE image_id = " + "'" + str(id) + "'")
            data = COLMAPDatabase.blob_to_array(data.fetchone()[0], np.uint8)
            descs_rows = int(np.shape(data)[0] / 128)
            descs = data.reshape([descs_rows, 128]) #descs for the whole image
            keypoint_index = points3D[point_id].point2D_idxs[k]
            desc = descs[keypoint_index] #keypoints and descs are ordered the same (so I use the point2D_idxs to index descs )
            desc = desc.reshape(1, 128) #this is the desc of keypoint with index, keypoint_index, from image with id, id.
            desc = desc / desc.sum()
            points3D_descs = np.r_[points3D_descs, desc]

        # adding and calulating the mean here!
        points_mean_descs = np.r_[points_mean_descs, points3D_descs.mean(axis=0).reshape(1,128)]
    return points_mean_descs

base_path = sys.argv[1]
db_path = os.path.join(base_path,"database.db")
model_images_bin_path = os.path.join(base_path,"model/0/images.bin")
model_points3D_bin_path = os.path.join(base_path,"model/0/points3D.bin")
save_path = os.path.join(base_path,"model/0/avg_descs.npy")

db = COLMAPDatabase.connect(db_path)
images = read_images_binary(model_images_bin_path)
points3D = read_points3d_default(model_points3D_bin_path)

avgs = get_desc_avg(points3D, db)
np.save(save_path, avgs)
