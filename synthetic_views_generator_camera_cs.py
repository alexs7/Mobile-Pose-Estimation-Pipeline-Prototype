# This file will use save camera 3D points and their 2D corresponding coordinates
# A camera pose can convert from world space points to camera 3D points (for example, COLMAP).
# It uses the depth info to get the 3D points.

import os
import sys
import time

import cv2
import numpy as np
import open3d as o3d

np.set_printoptions(precision=3, suppress=True)

from cyens_database import CYENSDatabase

WIDTH = 1920
HEIGHT = 1080

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

def getPoints(cams_csv):
    points = np.empty([0, 3])
    for cam in cams_csv:
        cam_center_cx = cam[1]  # world
        cam_center_cy = cam[2]
        cam_center_cz = cam[3]
        point = np.array([cam_center_cx, cam_center_cy, cam_center_cz])
        points = np.r_[ points, np.reshape(point, [1, 3])]
    return points

def create_cams_from_bundler(bundler_data, cams_csv):
    # The params below were generated by creating a window of the same height and width as below, and
    # then setting the camera manually at a point in space (you can just leave it at starting position)
    # then you save the params with write_pinhole_camera_parameters() and read them.
    h = 1080
    w = 1920
    f = 935.3
    px = 959.5
    py = 539.5
    bundler_cams = []
    for i in range(3, len(bundler_data), 5):
        if( i >= len(cams_csv) * 5 ):
            break
        k = i
        r1 = np.fromstring(bundler_data[k], sep=" ")
        r2 = np.fromstring(bundler_data[k+1], sep=" ")
        r3 = np.fromstring(bundler_data[k+2], sep=" ")
        t = np.fromstring(bundler_data[k+3], sep=" ")
        rotm = np.array([r1, r2, r3])
        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, f, f, px, py)
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = intrinsics
        # the camera here is given in camera coordinate system, https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
        extrinsics = np.r_[np.c_[rotm, t], np.array([0, 0, 0, 1]).reshape(1, 4)]
        cam_params.extrinsic = extrinsics
        bundler_cams.append(cam_params)
        breakpoint()
    return bundler_cams

def create_trajectory(base_path):
    cams_path = os.path.join(base_path, "model_files/Internal_ExternalCameraParameters/Internal_external.csv")
    bundler_file_path_right_handed = os.path.join(base_path, "model_files/BUNDLER/bundler_poses_negative_z_axis_right_handed.out")

    with open(bundler_file_path_right_handed) as f:
        bundler_file_right_handed = f.readlines()

    cams_csv = np.loadtxt(cams_path, dtype='object, float, float, float, float, float, float, float, float, float, float, float, float, float', usecols=(range(0, 14)), delimiter=',')
    cams_bundler = create_cams_from_bundler(bundler_file_right_handed, cams_csv)

    trajectory_cams = []
    for cam in cams_bundler:
        extrinsics = cam.extrinsic  # in camera coordinates

        rot_fix_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # set to world coordinates
        inverse = np.linalg.inv(extrinsics)
        rotm_inv = inverse[0:3, 0:3]
        # flip around the x-axis 180
        rotm_inv = np.matmul(rotm_inv, rot_fix_1)
        trans_inv = inverse[0:3, 3]
        extrinsics = np.r_[np.c_[rotm_inv, trans_inv], np.array([0, 0, 0, 1]).reshape(1, 4)]

        # reset back to camera coordinates
        extrinsics = np.linalg.inv(extrinsics)

        # TODO: Keep this code as it was used to visualise the camera poses when loading the mesh
        # You will need to .add(cam_vis) for each camera
        # rotm = extrinsics[0:3, 0:3]
        # trans = extrinsics[0:3, 3]
        #
        # takes the pose in camera coordinates (does an inverse in it)
        # cam_vis = o3d.geometry.LineSet.create_camera_visualization(cam.intrinsic.width, cam.intrinsic.height, cam.intrinsic.intrinsic_matrix, extrinsics)
        #
        # cam_vis_coor_sys = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=trans_inv)
        # cam_vis_coor_sys.rotate(rotm_inv)

        trajectory_cam = o3d.camera.PinholeCameraParameters()
        trajectory_cam.intrinsic = cam.intrinsic
        trajectory_cam.extrinsic = extrinsics  # save the pose in camera coordinates
        trajectory_cams.append(trajectory_cam)

    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = trajectory_cams

    return trajectory

def custom_draw_geometry_with_camera_trajectory(mesh, trajectory, base_path, width, height):
    vis = o3d.visualization.Visualizer()
    sift = cv2.SIFT_create()

    synth_images_path = os.path.join(base_path, "synth_images/")
    depths_path = os.path.join(base_path, "depths/")
    poses_path = os.path.join(base_path, "poses/")
    verifications_path = os.path.join(base_path, "verifications/")

    if not os.path.exists(synth_images_path):
        os.makedirs(synth_images_path)
    if not os.path.exists(depths_path):
        os.makedirs(depths_path)
    if not os.path.exists(poses_path):
        os.makedirs(poses_path)
    if not os.path.exists(verifications_path):
        os.makedirs(verifications_path)

    vis.create_window(width=width, height=height)
    vis.add_geometry(mesh)

    data_length = len(trajectory.parameters)
    depth_scale = 1000
    row_length = 134
    print("Data size: " + str(data_length))

    for i in range(data_length):
        print("Setting up data for pose: " + str(i))
        # in camera coordinates
        pose = trajectory.parameters[i]
        ctr = vis.get_view_control()
        # will convert to world coordinates here (I found no way to extract the
        # world pose from this method, maybe you can just get it from the camera pose? - not really I couldn't )
        ctr.convert_from_pinhole_camera_parameters(pose, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()

        synth_image_path = os.path.join(synth_images_path, "{:05d}.png".format(i))
        depth_path = os.path.join(depths_path, "{:05d}.png".format(i))
        depth_float_path = os.path.join(depths_path, "{:05d}_float.npy".format(i))
        verifications_image_path = os.path.join(verifications_path, "{:05d}.png".format(i))

        # save synth image
        vis.capture_screen_image(synth_image_path)

        # save both depths
        vis.capture_depth_image(depth_path)
        depth_float = vis.capture_depth_float_buffer() #returns an image is saved as numpy array though
        np.save(depth_float_path, depth_float)

        # save camera poses
        captured_poses_path = os.path.join(poses_path, "{:05d}.json".format(i))
        o3d.io.write_pinhole_camera_parameters(captured_poses_path, pose) # save the pose in camera coordinates

        print(" Extracting data from pose: " + str(i))

        synth_image = cv2.imread(synth_image_path)

        depth_path = os.path.join(depths_path, "{:05d}.png".format(i))
        depth_map = cv2.imread(depth_path, cv2.CV_16U) # if use this then divide by 1000
        depth_float_map = np.load(depth_float_path)
        depth_map_for_mask = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        mask = np.copy(depth_map_for_mask)  # this mask will be used for OpenCV
        mask[np.where(mask > 0)] = 255

        kps_train, descs_train = sift.detectAndCompute(synth_image, mask=mask)

        print(" Drawing detected keypoints on image.. (green)")
        kps_train_xy = cv2.KeyPoint_convert(kps_train)
        synth_image_verification = synth_image.copy()

        for j in range(len(kps_train_xy)):
            xy_drawing = np.round(kps_train_xy[j]).astype(int)
            cv2.circle(synth_image_verification, (xy_drawing[0], xy_drawing[1]) , 4, (0, 255, 0), -1)

        intrinsics = pose.intrinsic.intrinsic_matrix
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        extrinsics = pose.extrinsic

        print(" Estimating 3D point coordinates (in camera space) using the depth maps.")
        data_rows = np.empty([0, row_length])
        for k in range(len(kps_train)):
            keypoint = kps_train[k]
            descriptor = descs_train[k]  # same order as above
            xy = np.round(keypoint.pt).astype(int)  # openCV convention (x,y)
            # numpy convention
            row = xy[1]
            col = xy[0]
            depth = depth_float_map[row, col]
            # z = depth / depth_scale  # if you use vis.capture_depth_float_buffer() to get the depth do not divide by depth_scale
            z = depth
            x = (xy[0] - cx) * z / fx
            y = (xy[1] - cy) * z / fy

            # the points here x,y,z are in camera coordinates
            point_camera_coordinates = np.array([x, y, z, 1]).reshape([4, 1])

            # projecting the camera points (camera coordinate system) on the frame
            points_projected = pose.intrinsic.intrinsic_matrix.dot(point_camera_coordinates[0:3])
            points_projected = points_projected / points_projected[2, :]
            points_projected = np.round(points_projected.transpose()).astype(int)[0]

            assert(points_projected[0] == xy[0])
            assert(points_projected[1] == xy[1])

            data_row = np.append(np.array([xy[0], xy[1], x, y, z, depth]), descriptor).reshape([1, row_length])
            data_rows = np.r_[data_rows, data_row]

            # pass (x,y) but reverses them in the method to (y,x)
            cv2.circle(synth_image_verification, (points_projected[0], points_projected[1]) , 3, (0, 0, 255), -1)

        print(" Saving verification image..")
        cv2.imwrite(verifications_image_path, synth_image_verification)
        print(" Saving db row..")
        db.add_feature_data(i, data_rows)

    vis.destroy_window()

base_path = sys.argv[1] # i.e. /Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/
database_path = os.path.join(base_path, "features_data_ccs.db")

start = time.time()

print("Creating database...")
db = CYENSDatabase.connect(database_path)
db.create_data_table()

print("Loading objects...")
mesh_path = os.path.join(base_path, "model_files/EXPORT_Mesh/model.fbx")

print("Reading mesh...")
mesh = o3d.io.read_triangle_mesh(mesh_path)

print("Creating trajectory...")
trajectory = create_trajectory(base_path)

print("Traversing trajectory...")
custom_draw_geometry_with_camera_trajectory(mesh, trajectory, base_path, WIDTH, HEIGHT)

print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))

exit()

# Old code:

# poses = pnp(pts_2d=correspondences_2D_3D[:,0:2],
#             pts_3d=correspondences_2D_3D[:,2:5],
#             K=pose.intrinsic.intrinsic_matrix,
#             max_iters = 5000)
# R, t = poses[0]
# est_pose = np.r_[(np.c_[R, t]), [np.array([0, 0, 0, 1])]]

# print("Projecting world points on image using intrinsics + world pose matrix.. (red)")
#
# keypoints_world_points_3D
# keypoints_world_points_3D = np.hstack((keypoints_world_points_3D, np.ones((keypoints_world_points_3D.shape[0], 1))))
# points_2D_world_projected = pose.intrinsic.intrinsic_matrix.dot(est_pose.dot(keypoints_world_points_3D.transpose())[0:3, :])
# points_2D_world_projected = points_2D_world_projected // points_2D_world_projected[2, :]
# points_2D_world_projected = points_2D_world_projected.transpose()
# points_2D_world_projected = points_2D_world_projected.astype(int)

# for point in points_2D_world_projected:
#     cv2.circle(synth_image_verification, (point[0], point[1]) , 2, (0, 0, 255), -1)

# print("Estimating 3D point coordinates (in camera space) using the depth maps.")
# data_rows = np.empty([0, row_length])
# point_world_coordinates_all = np.empty([0,3])
# for k in range(len(kps_train)):
#     keypoint = kps_train[k]
#     descriptor = descs_train[k]  # same order as above
#     xy = np.round(keypoint.pt).astype(int)  # openCV convention (x,y)
#     # numpy convention
#     row = xy[1]
#     col = xy[0]
#     depth = depth_float_map[row, col]
#     # z = depth / depth_scale  # if you use vis.capture_depth_float_buffer() to get the depth do not divide by depth_scale
#     z = depth
#     x = (xy[0] - cx) * z / fx
#     y = (xy[1] - cy) * z / fy
#
#     # the points here x,y,z are in camera coordinates
#     point_camera_coordinates = np.array([x, y, z, 1]).reshape([4, 1])
#
#     # projecting the camera points (camera coordinate system) on the frame
#     points_projected = pose.intrinsic.intrinsic_matrix.dot(point_camera_coordinates[0:3])
#     points_projected = points_projected // points_projected[2, :]
#     points_projected = points_projected.transpose().astype(int)[0]
#
#     cv2.circle(synth_image_verification, (points_projected[0], points_projected[1]) , 2, (0, 0, 0), -1)
#
#     rotm_t = extrinsics[0:3, 0:3].transpose()
#     trans = -rotm_t.dot(extrinsics[0:3, 3])
#     cam_to_world = np.r_[np.c_[rotm_t, trans], np.array([0, 0, 0, 1]).reshape(1,4)]
#
#     # transform camera points to world points with inv(pose)
#     # TODO: divide by z here ?
#     point_world_coordinates = cam_to_world.dot(point_camera_coordinates)
#     point_world_coordinates = point_world_coordinates[0:3, :]
#     point_world_coordinates_all = np.r_[point_world_coordinates_all, point_world_coordinates.reshape(1,3)]
#
#     x_world = point_world_coordinates[0,0]
#     y_world = point_world_coordinates[1,0]
#     z_world = point_world_coordinates[2,0]
#     data_row = np.append(np.array([xy[0], xy[1], x_world, y_world, z_world, depth]), descriptor).reshape([1, row_length])
#     data_rows = np.r_[data_rows, data_row]

# cv2.imwrite(synth_image_verification_path, synth_image_verification)

# print("Creating point clouds..")
#
# print("  Point cloud 1/3")
# # will be used many times
# pointcloud_verification = o3d.geometry.PointCloud()
# colors = [[0, 0.6, 0] for i in range(point_world_coordinates_all.shape[0])]
# pointcloud_verification.colors = o3d.utility.Vector3dVector(colors)
# pointcloud_verification.points = o3d.utility.Vector3dVector(point_world_coordinates_all)
# vis.add_geometry(pointcloud_verification)
#
# print("  Point cloud 2/3")
# pointcloud_verification = o3d.geometry.PointCloud()
# point_world_coordinates_all_point_cloud_world_estimated = correspondences_2D_3D[:,2:5]
# colors = [[0.6, 0, 0] for i in range(point_world_coordinates_all_point_cloud_world_estimated.shape[0])]
# pointcloud_verification.colors = o3d.utility.Vector3dVector(colors)
# pointcloud_verification.points = o3d.utility.Vector3dVector(point_world_coordinates_all_point_cloud_world_estimated)
# vis.add_geometry(pointcloud_verification)

# print("  Point cloud 3/3")
# pointcloud_verification = o3d.geometry.PointCloud()
# colors = [[0, 0, 0.6] for i in range(np.asarray(debug_point_cloud_world.points).shape[0])]
# pointcloud_verification.colors = o3d.utility.Vector3dVector(colors)
# pointcloud_verification.points = o3d.utility.Vector3dVector(debug_point_cloud_world.points)
# vis.add_geometry(pointcloud_verification)

# ctr = vis.get_view_control()
# ctr.convert_from_pinhole_camera_parameters(pose, allow_arbitrary=True)
# vis.poll_events()
# vis.update_renderer()