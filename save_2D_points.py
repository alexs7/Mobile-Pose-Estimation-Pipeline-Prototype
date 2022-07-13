import cv2
import numpy as np

def save_projected_points(points_3D, keypoints_2D, est_pose_query, K, real_img, verification_image_path):
    green = (0, 255, 0)
    blue = (255, 0, 0)
    image = real_img.copy()
    points_3D = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))
    points = K.dot(est_pose_query.dot(points_3D.transpose())[0:3,:])
    points = points // points[2,:]
    points = points.transpose()
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        x_real = int(keypoints_2D[i][0])
        y_real = int(keypoints_2D[i][1])
        center = (x, y)
        center_real = (x_real, y_real)
        cv2.circle(image, center_real, 14, green, -1)
        cv2.circle(image, center, 12, blue, -1)
    cv2.imwrite(verification_image_path, image)