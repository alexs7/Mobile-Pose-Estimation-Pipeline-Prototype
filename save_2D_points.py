import cv2
import numpy as np
from  sklearn.metrics import mean_squared_error

def save_points_2D_2D(points_2D_est, points_2D_gt, image, path):
    green = (0, 255, 0)
    blue = (255, 0, 0)
    for i in range(len(points_2D_est)):
        x = int(points_2D_est[i][0])
        y = int(points_2D_est[i][1])
        x_real = int(points_2D_gt[i][0])
        y_real = int(points_2D_gt[i][1])
        center = (x, y)
        center_real = (x_real, y_real)
        cv2.circle(image, center_real, 4, green, -1)
        cv2.circle(image, center, 3, blue, -1)
    cv2.imwrite(path, image)

def save_projected_points(points_3D, keypoints_2D, est_pose_query,
                          K, real_img, verification_image_path):
    green = (0, 255, 0) # for the good matches keypoints
    blue = (255, 0, 0) # for the projected 3D points
    image = real_img.copy()
    points_3D = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))
    points = K.dot(est_pose_query.dot(points_3D.transpose())[0:3,:])
    points = points / points[2,:]
    # Note that some points will not show up because they are outliers.
    # To visually check this: look at the matches from the query image to the
    # synth image. Some query points are matched to wrong synth points.
    # RANSAC sees them as outliers and discards them, that is why some no blue points over green
    points = points.transpose()

    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        x_real = int(keypoints_2D[i][0])
        y_real = int(keypoints_2D[i][1])
        center = (x, y)
        center_real = (x_real, y_real)
        cv2.circle(image, center_real, 4, green, -1)
        cv2.circle(image, center, 3, blue, -1)
    cv2.imwrite(verification_image_path, image)