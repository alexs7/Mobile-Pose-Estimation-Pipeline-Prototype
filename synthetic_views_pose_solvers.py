import pycolmap
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation as R
import cv2

def pycolmap_wrapper_absolute_pose(points_2D, points_3D, camera, max_error_px = 3.0):
    pose = pycolmap.absolute_pose_estimation(points_2D, points_3D, camera, max_error_px = max_error_px)
    quat = pose['qvec']
    tvec = pose['tvec']
    rotM = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix() # expects [x, y, z, w]
    pose = np.c_[rotM, tvec]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

def opencv_pose(points_2D, points_3D, K, flag = cv2.SOLVEPNP_P3P):
    _, rvec, tvec, _ = cv2.solvePnPRansac(points_3D.astype(np.float32), points_2D.astype(np.float32),
                                          K, distCoeffs=None, iterationsCount=3000,
                                          confidence=0.99, flags = flag)
    rotM = cv2.Rodrigues(rvec)[0]
    pose = np.c_[rotM, tvec]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

def project_points(points_3D, K, pose):
    if points_3D.shape[1] == 3:
        points_3D = add_ones(points_3D)
    #2D points
    points = K.dot(pose.dot(points_3D.transpose())[0:3, :])
    points = points / points[2, :]
    points = points.transpose()
    return points

def add_ones(points):
    return np.c_[points, np.ones(points.shape[0])]

