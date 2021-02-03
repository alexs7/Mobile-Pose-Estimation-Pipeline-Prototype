# This is to view the keypoints detected on the localised frame and project the points from the 3D model and compare
from query_image import read_images_binary
from show_2D_points import print_points_on_image
import numpy as np

path_images_new_model = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/alfa_mega/2020-12-20/model/0/images.bin"
images_new = read_images_binary(path_images_new_model)

for k,v in images_new.items():
    if(v.name == "frame_1608461089615.jpg"):
        print_points_on_image("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/all_data_and_models/alfa_mega/2020-12-20/arcore/frame_1608461089615.jpg", v.xys, (0, 0, 255), "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/debug_image.jpg")
        # points2D = np.empty([0,2])
        # for i in range(len(v.point3D_ids)):
        #     if(v.point3D_ids[i] != -1):
        #         points2D = np.r_[points2D, v.xys[i].reshape([1, 2])]

        # print("Visible 3D points no: " + str(len(points2D)))
        # show_projected_points("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/detected_keypoints.jpg", points2D, (0, 0, 255), "projected_3D_points.jpg")
        break