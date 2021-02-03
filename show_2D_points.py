from random import random, randint, randrange, sample

import cv2
import numpy as np
from point3D_loader import read_points3d_default

# K = np.loadtxt("matrices/pixel_intrinsics_low_640.txt")

# live_model_points3D_path = "/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/new_model/points3D.bin"
# points3D = read_points3d_default(live_model_points3D_path)

def show_projected_points(image_path, sorted_matches, sample, output):
    np.save("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/sorted_matches.npy", sorted_matches)
    lowes_distances = sorted_matches[:, 6] / sorted_matches[:, 6].sum()
    heatmap_vals = sorted_matches[:, 7] / sorted_matches[:, 7].sum()
    points2D_all = sorted_matches[:, 0:2]
    points2D_sample = sample[:, 0:2]
    red = (0, 0, 255)
    blue = (255, 0, 0)
    image = cv2.imread(image_path)
    for i in range(int(len(points2D_all))):
        x = int(points2D_all[i][0])
        y = int(points2D_all[i][1])
        center = (x, y)
        cv2.circle(image, center, 2, blue, -1)
    #     also add text
        val = lowes_distances[i] * heatmap_vals[i]
        # cv2.putText(image, "{:.10f}".format(val), (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255, 255, 255), thickness = 2 )
    for i in range(len(points2D_sample)):
        val = lowes_distances[i] * heatmap_vals[i]
        # print("sample's val :" + "{:.10f}".format(val))
        # print("sample's point3D id :" + str(sample[i,5]))
        x = int(points2D_sample[i][0])
        y = int(points2D_sample[i][1])
        # print("sample's x and y :" + str(x) + " " + str(y))
        center = (x, y)
        cv2.circle(image, center, 5, red, -1)
    cv2.imwrite("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/"+output, image)

def print_points_on_image(image_path, points, color, output_path):
    image = cv2.imread(image_path)
    random_list = sample(range(0, 2600), 100)
    for i in random_list:
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 3, color, -1)
    cv2.imwrite(output_path, image)