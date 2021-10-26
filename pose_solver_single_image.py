import numpy as np

from ransac_prosac import ransac

def solve(matches, intrinsics):

    assert(len(matches) >= 4)
    best_model = ransac(matches, intrinsics)
    assert(best_model != None)

    pose = best_model['Rt']
    return pose

def apply_transform(colmap_pose, arcore_pose, scale, points3D_xyz_rgb):
    colmap_to_arcore_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0]])
    colmap_to_arcore_matrix = scale * colmap_to_arcore_matrix
    colmap_to_arcore_matrix = np.r_[colmap_to_arcore_matrix, [np.array([0, 0, 0, 1])]]

    rotZ = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    intermediate_matrix = rotZ.dot(colmap_to_arcore_matrix)

    # from_colmap_world_to_colmap_camera
    points3D = colmap_pose.dot(np.transpose(points3D_xyz_rgb[:,0:4])) #homogeneous
    # from_colmap_camera_to_arcore_camera
    points3D = intermediate_matrix.dot(points3D)
    # from_arcore_camera_to_arcore_world
    points3D = (arcore_pose.dot(points3D))
    points3D = np.transpose(points3D)

    points3D_xyz_rgb_transformed = np.c_[points3D, points3D_xyz_rgb[:,4:7]] #xyz , 1, rgb
    return points3D_xyz_rgb_transformed