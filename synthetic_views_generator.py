import sys
import time
from os.path import join
import numpy as np
import open3d as o3d
import os
import cv2

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
        extrinsics = np.r_[np.c_[rotm, t], np.array([0, 0, 0, 1]).reshape(1, 4)]
        cam_params.extrinsic = extrinsics
        bundler_cams.append(cam_params)
    return bundler_cams

def custom_draw_geometry_with_camera_trajectory(mesh, trajectory, base_path):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory = trajectory
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()

    image_path = os.path.join(base_path, "image/")
    depth_path = os.path.join(base_path, "depth/")
    poses_path = os.path.join(base_path, "poses/")

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    if not os.path.exists(poses_path):
        os.makedirs(poses_path)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            print("Capturing image {:05d}..".format(glb.index))
            captured_image_path = os.path.join(image_path, "{:05d}.png".format(glb.index))
            captured_depth_path = os.path.join(depth_path, "{:05d}.png".format(glb.index))
            vis.capture_depth_image(captured_depth_path, False)
            vis.capture_screen_image(captured_image_path, False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            print("Saving pose {:05d}..".format(glb.index))
            pose = glb.trajectory.parameters[glb.index] # camera parameters
            ctr.convert_from_pinhole_camera_parameters(pose)
            captured_poses_path = os.path.join(poses_path, "{:05d}.json".format(glb.index))
            o3d.io.write_pinhole_camera_parameters(captured_poses_path, pose)
        else:
            custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(mesh)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()

base_path = sys.argv[1] # i.e. /Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/

start = time.time()

print("Loading objects...")
cams_path = os.path.join(base_path, "model_files/Internal_ExternalCameraParameters/Internal_external_1st_Model.csv")
mesh_path = os.path.join(base_path, "model_files/EXPORT_Mesh/1st_MODEL_-_4k_Video_Photogrammetry.fbx")
imgs_path = os.path.join(base_path, "IMAGES/C0002 frames")
bundler_file_path_right_handed = os.path.join(base_path, "1st MODEL - FILES/BUNDLER/1st_MODEL_-_4k_Video_Photogrammetry.out")

# switch file here
with open(bundler_file_path_right_handed) as f:
    bundler_file_right_handed = f.readlines()

cams_csv = np.loadtxt(cams_path, dtype='object, float, float, float, float, float, float, float, float, float, float, float, float, float', usecols=(range(0,14)), delimiter=',')
cams_bundler = create_cams_from_bundler(bundler_file_right_handed, cams_csv)

print("Reading mesh...")
mesh = o3d.io.read_triangle_mesh(mesh_path)

print("Creating trajectory...")
trajectory_cams = []
for cam in cams_bundler:
    extrinsics = cam.extrinsic #in camera coordinates

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

    rotm = extrinsics[0:3, 0:3]
    trans = extrinsics[0:3,3]

    cam_vis = o3d.geometry.LineSet.create_camera_visualization(cam.intrinsic.width, cam.intrinsic.height,
                                                               cam.intrinsic.intrinsic_matrix, extrinsics)

    cam_vis_coor_sys = o3d.geometry.TriangleMesh.create_coordinate_frame(origin = trans_inv)
    cam_vis_coor_sys.rotate(rotm_inv)

    trajectory_cam = o3d.camera.PinholeCameraParameters()
    trajectory_cam.intrinsic = cam.intrinsic
    trajectory_cam.extrinsic = extrinsics
    trajectory_cams.append(trajectory_cam)

trajectory = o3d.camera.PinholeCameraTrajectory()
trajectory.parameters = trajectory_cams

print("Traversing trajectory...")
custom_draw_geometry_with_camera_trajectory(mesh, trajectory, base_path)

print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))