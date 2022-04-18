import numpy as np
import cv2
from query_image import get_query_image_global_pose
from point3D_loader import get_points3D
from query_image import get_query_image_id
import glob
import scipy.io as sio
from scipy.spatial.transform import Rotation as R
import os

K = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/matrices/pixel_intrinsics_low_640_landscape.txt")

def get_pose_from_correspondences(filename,K):
    correspondences = np.loadtxt(filename)
    (_, rvec, tvec, inliers_direct) = cv2.solvePnPRansac(correspondences[:,2:5], correspondences[:,0:2], K, None, iterationsCount = 500, confidence = 0.99, flags = cv2.SOLVEPNP_EPNP)
    rotM = cv2.Rodrigues(rvec)[0]
    pose = np.c_[rotM, tvec]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

def show_projected_points(image_path, K, FP, points3D):
    image = cv2.imread(image_path)
    points = K.dot(FP.dot(points3D.transpose())[0:3,:])
    points = points // points[2,:]
    points = points.transpose()
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 0, 255), -1)
    cv2.imshow("result", image)
    cv2.waitKey(0)

def show_projected_points_only_intrinsics(image_path, K, Rt_points3D):
    image = cv2.imread(image_path)
    Rt_points3D = Rt_points3D[:, 0:3]
    points = K.dot(Rt_points3D.transpose())
    points = points // points[2,:]
    points = points.transpose()
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 0, 255), -1)
    cv2.imshow("result", image)
    cv2.waitKey(0)

def save_projected_points_only_intrinsics(image_path, K, Rt_points3D):
    image = cv2.imread(image_path)
    Rt_points3D = Rt_points3D[:, 0:3]
    points = K.dot(Rt_points3D.transpose())
    points = points // points[2,:]
    points = points.transpose()
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 0, 255), -1)
    cv2.imwrite('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/debug_projected.jpg',image)

def save_image_projected_points(image_path, K, FP, points3D):
    image = cv2.imread(image_path)
    points = K.dot(FP.dot(points3D.transpose())[0:3,:])
    points = points // points[2,:]
    points = points.transpose()
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 4, (0, 0, 255), -1)
    cv2.imwrite('/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/colmap_points_projected.jpg', image)

def save_image_projected_points_unity(image_path, K, FP, points3D, output_path):
    image = cv2.imread(image_path)
    points = K.dot(FP.dot(points3D.transpose())[0:3,:])
    points = points // points[2,:]
    points = points.transpose()
    for i in range(len(points)):
        x = int(points[i][0])
        y = int(points[i][1])
        center = (x, y)
        cv2.circle(image, center, 2, (0, 0, 255), -1)
    cv2.imwrite(output_path, image)

def get_ARCore_displayOrientedPose(dir, frame):
    return np.loadtxt(dir+"/displayOrientedPose_"+frame+".txt")

def get_ARCore_poses(dir, pose_string, sequence):
    print("Getting ARCore poses..")
    poses = []
    for i in range(len(sequence)):
        poses.append(np.loadtxt(dir + "/" + pose_string+'_'+sequence[i]+'.txt'))
    return poses

def get_ARCore_pose_query_image(path):
    f = open(path, 'r')
    x = f.readlines()
    f.close()
    values = x[0].split(',')
    tx = float(values[0])
    ty = float(values[1])
    tz = float(values[2])
    qx = float(values[3])
    qy = float(values[4])
    qz = float(values[5])
    qw = float(values[6])
    quat = [qx, qy, qz, qw] # This already comes normalised - I checked the norm with matlab
    rot = R.from_quat(quat)
    rot = rot.as_dcm()
    tvec = np.array([tx, ty, tz])
    pose = np.c_[rot, tvec]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

def get_Unity_pose_query_image(path):
    f = open(path, 'r')
    x = f.readlines()
    f.close()
    values = x[0].split(',')
    tx = float(values[0])
    ty = float(values[1])
    tz = float(values[2])
    qx = float(values[3])
    qy = float(values[4])
    qz = float(values[5])
    qw = float(values[6])
    quat = [qx, qy, qz, qw]  # This already comes normalised - I checked the norm with matlab
    rot = R.from_quat(quat)
    rot = rot.as_dcm()
    tvec = np.array([tx, ty, tz])
    pose = np.c_[rot, tvec]
    pose = np.r_[pose, [np.array([0, 0, 0, 1])]]
    return pose

def get_ARCore_pose_query_image_matrix_file():
    pose = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/query_data/cameraPoseMatrixString.txt")
    return pose

def get_ARCore_poses_relative(poses):
    print("Getting ARCore Relative poses..")
    relative_poses = []
    start = poses[0]

    relative_poses.append(start)

    relative_pose = np.empty([4, 4])
    for i in range(1, len(poses)):
        lp = poses[i]

        R1 = start[0:3, 0:3]
        t1 = start[0:3, 3]
        R2 = lp[0:3, 0:3]
        t2 = lp[0:3, 3]
        R1to2 = np.linalg.inv(R2).dot(R1)
        T1to2 = np.linalg.inv(R2).dot((t1 - t2))

        relative_pose[0:3, 0:3] = R1to2
        relative_pose[0:3, 3] = T1to2
        relative_pose[3, :] = [0, 0, 0, 1]

        fp = relative_pose.dot(start)
        relative_poses.append(fp)

    return relative_poses

def get_sequence(dir, pose_string, start, end):
    sequence = []
    for filename in glob.glob(dir+"/"+pose_string+'_*.txt'): #can be anything
        sequence.append(filename.split("_")[-1].split(".")[0])
    sequence  = np.array(sequence)
    sequence.sort()
    start_index = np.where(sequence == start)[0][0]
    end_index = np.where(sequence == end)[0][0]
    sequence = sequence[start_index: end_index + 1]
    return sequence

def get_sequence_all(dir):
    sequence = []
    for filename in glob.glob(dir + '/displayOrientedPose_*.txt'):  # can be anything
        sequence.append(filename.split("_")[-1].split(".")[0])
    sequence = np.array(sequence)
    sequence.sort()
    return sequence


def get_COLMAP_poses(array):
    print("Getting COLMAP poses..")
    poses = []
    for i in range(len(array)):
        print("At: " + str(round(100 * i / len(array),2)) + "%", end="\r")
        poses.append(get_query_image_global_pose("frame_" + array[i] + ".jpg"))
    return poses

def get_Relative_Poses(local_poses, global_poses):
    print("Getting Relative poses..")
    relative_poses = []
    gp_start = global_poses[0] #just use first anyway
    lp_start = local_poses[0]

    fp = lp_start.dot(np.linalg.inv(lp_start)).dot(gp_start)
    relative_poses.append(fp)

    relative_pose = np.empty([4, 4])
    for i in range(1, len(local_poses)):
        lp = local_poses[i]

        R1 = lp_start[0:3, 0:3]
        t1 = lp_start[0:3, 3]
        R2 = lp[0:3, 0:3]
        t2 = lp[0:3, 3]
        R1to2 = np.linalg.inv(R2).dot(R1)
        T1to2 = np.linalg.inv(R2).dot((t1 - t2))

        relative_pose[0:3, 0:3] = R1to2
        relative_pose[0:3, 3] = T1to2
        relative_pose[3, :] = [0, 0, 0, 1]

        fp = relative_pose.dot(gp_start)
        relative_poses.append(fp)

    return relative_poses

def save_poses(poses, folder_name):
    for i in range(len(poses)):
        np.savetxt(folder_name+"/pose_" + str(i) + ".txt", poses[i])

# make sure this is commented out when localising from Electron
# check and load the files here
# also check the trajectory if it didnt break in ARCore
# query_dir = '/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/images/data_ar'
# sequence = get_sequence(query_dir, 'displayOrientedPose', '1582476378648' , '1582476394307')
#
# ARCore_oriented_poses = get_ARCore_poses(query_dir, 'displayOrientedPose', sequence)
# ARCore_poses_relative = get_ARCore_poses_relative(ARCore_oriented_poses)
# ARCore_camera_poses = get_ARCore_poses(query_dir, 'cameraPose', sequence)
# COLMAP_poses = get_COLMAP_poses(sequence)
# ar_core_and_colmap_relative_poses = get_Relative_Poses(ARCore_oriented_poses, COLMAP_poses)
#
# # os.system("rm poses_for_debug/ar_core_oriented_poses/*")
# # os.system("rm poses_for_debug/arcore_relative_poses/*")
# # os.system("rm poses_for_debug/ar_core_camera_poses/*")
# # os.system("rm poses_for_debug/colmap_poses/*")
# # os.system("rm poses_for_debug/ar_core_and_colmap_relative_poses/*")
#
# save_poses(ARCore_oriented_poses, "poses_for_debug/ar_core_oriented_poses") # yellow in MATLAB
# save_poses(ARCore_poses_relative, "poses_for_debug/arcore_relative_poses") # green in MATLAB
# save_poses(ARCore_camera_poses, "poses_for_debug/ar_core_camera_poses") # white in MATLAB
# save_poses(COLMAP_poses, "poses_for_debug/colmap_poses") # red in MATLAB
# save_poses(ar_core_and_colmap_relative_poses, "poses_for_debug/ar_core_and_colmap_relative_poses") # blue in MATLAB


# old code
# CP_start = get_query_image_global_pose("frame_"+first_frame+".jpg")
#
# LP_start = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/query_data/cameraPose_" + first_frame + ".txt")
# FP_start = LP_start.dot(np.linalg.inv(LP_start)).dot(CP_start) # Identity anyway..
#
# image_id_start = get_query_image_id("frame_"+first_frame+".jpg")
# points3D_start = get_points3D(image_id_start)
#
# np.savetxt("points3D_start.txt", points3D_start)
#
# show_projected_points("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/current_query_image/frame_" + first_frame + ".jpg", K, FP_start, points3D_start)
#
# colmap_poses = np.empty([4,4,len(sequence)])
# for i in range(len(sequence)):
#     colmap_pose = get_query_image_global_pose("frame_"+sequence[i]+".jpg")
#     colmap_poses[:,:,i] = colmap_pose
#
# relative_pose = np.empty([4, 4])
# for i in range(1 , len(sequence)):
#
#     LP = np.loadtxt("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/query_data/cameraPose_"+sequence[i]+".txt")
#
#     R1 = LP_start[0:3,0:3]
#     t1 = LP_start[0:3, 3]
#     R2 = LP[0:3,0:3]
#     t2 = LP[0:3,3]
#     R1to2 = np.linalg.inv(R2).dot(R1)
#     T1to2 = np.linalg.inv(R2).dot((t1 - t2))
#
#     relative_pose[0:3, 0:3] = R1to2
#     relative_pose[0:3, 3] = T1to2
#     relative_pose[3, :] = [0, 0, 0, 1]
#
#     FP = relative_pose.dot(CP_start)
#
#     show_projected_points("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/current_query_image/frame_" + sequence[i] +".jpg", K, FP, points3D_start)
#
#     np.savetxt("final_poses/final_pose_" + sequence[i] + ".txt", FP)
#     np.savetxt("colmap_poses/colmap_pose_" + sequence[i] + ".txt", colmap_poses[: , : , i])

# relative_pose = np.empty([4, 4])
# for i in range(1 , len(sequence)):
#
#     FP = sio.loadmat("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/matlab_res/final_pose_"+sequence[i]+".txt.mat")
#     FP = np.array(FP.get('pose_new'))
#     # FP = relative_pose.dot(CP_start)
#
#     show_projected_points("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/current_query_image/frame_" + sequence[i] +".jpg", K, FP, points3D_start)
#
#     # np.savetxt("final_poses/final_pose_" + sequence[i] + ".txt", FP)
#     # np.savetxt("colmap_poses/colmap_pose_" + sequence[i] + ".txt", colmap_poses[: , : , i])


