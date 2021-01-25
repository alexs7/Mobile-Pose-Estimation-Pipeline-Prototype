import cv2
import numpy as np
from tensorflow import keras

import colmap
from database import COLMAPDatabase
from feature_matcher_single_image import get_image_id, get_keypoints_xy, get_queryDescriptors

db_path = "/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/all_data_and_models/alfa_mega/test_pics/database.db"
query_images_dir = "/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/all_data_and_models/alfa_mega/test_pics/arcore"
image_list_file = "/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/all_data_and_models/alfa_mega/test_pics/query_name.txt"

colmap.feature_extractor(db_path, query_images_dir, image_list_file)

db = COLMAPDatabase.connect(db_path)
model = keras.models.load_model("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/models/local/alfa_mega")

with open(image_list_file) as f:
    query_images = f.readlines()
query_images = [x.strip() for x in query_images]

for i in range(len(query_images)):
    q_img = query_images[i]
    image_id = get_image_id(db, q_img)
    # keypoints data
    keypoints_xy = get_keypoints_xy(db, image_id)
    queryDescriptors = get_queryDescriptors(db, image_id)

    predictions = model.predict(queryDescriptors)

    data = np.concatenate((keypoints_xy, predictions), axis=1)
    data = data[data[:, 2].argsort()[::-1]]

    np.savetxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/alfa_mega/test_pics/predictions/"+q_img.split(".")[0]+".txt", data)

    image = cv2.imread("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/alfa_mega/test_pics/arcore/"+q_img)
    top_points = data[0:50, 0:2]

    for i in range(int(len(top_points))):
        x = int(top_points[i][0])
        y = int(top_points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 0, 255), -1)

    cv2.imwrite("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/alfa_mega/test_pics/resulting_images/" + q_img, image)