# This file will create full size images of the mesh
# using the plain old Visualizer class does not work because it keeps the window at a set resolution
# 15/08/2022: Also added depth capture to get the pointcloud
import glob
import os
import sys
import time
import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ExifTags
from tqdm import tqdm

np.set_printoptions(precision=5, suppress=True)

WIDTH = 4233
HEIGHT = 2816

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

def create_cams_from_bundler(base_path, bundler_data, cams_csv):
    images_path = os.path.join(base_path, "model_files/EXACT_FRAGMENT_IMAGES")
    images_paths = glob.glob(os.path.join(images_path, "*.jpg"))
    images_paths.sort()

    # from https://ksimek.github.io/2013/08/13/intrinsic/
    # I know the focal length in mm is 30 or 20 from the metadata.
    # The sensor width is, 35mm full frame (35.6×23.8mm) from https://www.sony.co.uk/electronics/interchangeable-lens-cameras/ilce-7m3-body-kit/specifications
    focal_lengths_px = np.empty([0,2])
    for path in images_paths:
        im = Image.open(path)
        exif = { ExifTags.TAGS[k]: v for k, v in im._getexif().items() if k in ExifTags.TAGS }
        #TODO:  breakpoint() #Checkpoint try to get the EXIF from pycolmap and compare
        focal_length_mm = float(exif['FocalLength'])
        fx = focal_length_mm * exif['ExifImageWidth'] / 35.6
        fy = focal_length_mm * exif['ExifImageHeight'] / 23.8
        focal_lengths_px = np.r_[focal_lengths_px, np.array([fx, fy]).reshape(1,2)] #this array is assumed to be sorted with the bundler images

    h = 2816
    w = 4233
    # TODO: Double check that px/py ?
    px = 4233 / 2
    py = 2816 / 2
    bundler_cams = []
    focal_idx = 0
    for i in range(3, len(bundler_data), 5):
        if( i >= len(cams_csv) * 5 ):
            break
        k = i
        # f = np.fromstring(bundler_data[k-1], sep=" ")[0] # the bundler ones (f = fx, fy)
        fx = focal_lengths_px[focal_idx][0]
        fy = focal_lengths_px[focal_idx][1]
        r1 = np.fromstring(bundler_data[k], sep=" ")
        r2 = np.fromstring(bundler_data[k+1], sep=" ")
        r3 = np.fromstring(bundler_data[k+2], sep=" ")
        t = np.fromstring(bundler_data[k+3], sep=" ")
        rotm = np.array([r1, r2, r3])
        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, px, py)
        cam_params = o3d.camera.PinholeCameraParameters()
        cam_params.intrinsic = intrinsics
        # the camera here is given in camera coordinate system, https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.4-manual.html#S6
        extrinsics = np.r_[np.c_[rotm, t], np.array([0, 0, 0, 1]).reshape(1, 4)]
        cam_params.extrinsic = extrinsics
        bundler_cams.append(cam_params)
        focal_idx += 1

    assert(focal_idx == focal_lengths_px.shape[0]) #sanity check
    return bundler_cams

def create_trajectory(base_path):
    cams_path = os.path.join(base_path, "model_files/Internal_ExternalCameraParameters/Internal_external.csv")
    bundler_file_path_right_handed = os.path.join(base_path, "model_files/BUNDLER/bundler_poses_negative_z_axis_right_handed.out")

    with open(bundler_file_path_right_handed) as f:
        bundler_file_right_handed = f.readlines()

    cams_csv = np.loadtxt(cams_path, dtype='object, float, float, float, float, float, float, float, float, float, float, float, float, float', usecols=(range(0, 14)), delimiter=',')
    cams_bundler = create_cams_from_bundler(base_path, bundler_file_right_handed, cams_csv)

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

        trajectory_cam = o3d.camera.PinholeCameraParameters()
        trajectory_cam.intrinsic = cam.intrinsic
        trajectory_cam.extrinsic = extrinsics  # save the pose in camera coordinates
        trajectory_cams.append(trajectory_cam)

    trajectory = o3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = trajectory_cams

    return trajectory

def traverse_and_save_frames(mesh, trajectory, base_path, width, height):

    render = o3d.visualization.rendering.OffscreenRenderer(width=width, height=height)
    material = o3d.visualization.rendering.MaterialRecord()
    render.scene.add_geometry("mesh", mesh, material)

    synth_images_path = os.path.join(base_path, "synth_images/")
    depths_path = os.path.join(base_path, "depths/")

    if not os.path.exists(synth_images_path):
        os.makedirs(synth_images_path)
    if not os.path.exists(depths_path):
        os.makedirs(depths_path)

    for i in tqdm(range(len(trajectory.parameters))):
        print("Saving image {:03d} ..".format(i))
        pose = trajectory.parameters[i]
        render.setup_camera(pose.intrinsic, pose.extrinsic)
        # rgb image
        image = np.asarray(render.render_to_image())
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert Open3D RGB to OpenCV BGR
        synth_image_path = os.path.join(synth_images_path, "{:05d}.png".format(i))
        cv2.imwrite(synth_image_path,image)
        # depth image
        depth = np.asarray(render.render_to_depth_image(z_in_view_space=True))
        depth[depth == np.inf] = 0 #remove inf values
        depth_data_path = os.path.join(depths_path, "{:05d}.npy".format(i))
        np.save(depth_data_path, depth)

        # create pointcloud for debug
        fx = pose.intrinsic.get_focal_length()[0]
        fy = pose.intrinsic.get_focal_length()[1]
        cx = pose.intrinsic.get_principal_point()[0]
        cy = pose.intrinsic.get_principal_point()[1]
        # get each pixel from image
        pointcloud_debug_camera_points = []
        pointcloud_debug_camera_colors = []

        print("meshgrid approach")
        m, n = depth.shape
        y, x = np.mgrid[0:m, 0:n] # reverse y,x here because of conventions
        x_3D = (x - cx) * depth / fx
        y_3D = (y - cy) * depth / fy

        points_3D_TBR = np.dstack((np.dstack((x_3D, y_3D)), depth)).reshape(height * width, 3)
        points_3D_TBR[points_3D_TBR[:, 2] == 0] = 0 #clean up large invalid values

        print("iterative approach")
        for y in range(depth.shape[0]):
            for x in range(depth.shape[1]):
                z_it = depth[y, x]
                # if(z_it == np.inf):
                #     continue
                x_3D_it = (x - cx) * z_it / fx # x in 3D
                y_3D_it = (y - cy) * z_it / fy # y in 3D
                # the points here x,y,z are in camera coordinates
                point_camera_coordinates = np.array([x_3D_it, y_3D_it, z_it])
                color = np.flip(image[y,x]) / 255 #BRG to RGB
                pointcloud_debug_camera_colors.append(color)
                pointcloud_debug_camera_points.append(point_camera_coordinates)

        # use this code to visualise the point cloud of 1 frame
        # note that there will be gaps compared to the complete model because there is no depth value for the
        # points behind the pixels on the frame

        pointcloud_debug_camera_points_b = np.array(pointcloud_debug_camera_points)
        pointcloud_debug_camera_colors_b = np.array(pointcloud_debug_camera_colors)

        pointcloud_debug_camera_points = points_3D_TBR
        pointcloud_debug_camera_colors = np.flip((image.reshape(height * width, 3) / 255) , axis = 1)

        breakpoint()

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        pointcloud_verification = o3d.geometry.PointCloud()
        pointcloud_verification.colors = o3d.utility.Vector3dVector(pointcloud_debug_camera_colors)
        pointcloud_verification.points = o3d.utility.Vector3dVector(pointcloud_debug_camera_points)
        vis.add_geometry(pointcloud_verification)
        print("Added pointcloud")
        ctr = vis.get_view_control()
        pose.extrinsic = np.eye(4) #set to origin, since points are in camera space
        ctr.convert_from_pinhole_camera_parameters(pose, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        print("Updated renderer")
        vis.run()

    return None

base_path = sys.argv[1] # i.e. /Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/

start = time.time()
print("Loading objects...")
mesh_path = os.path.join(base_path, "model_files/FBX WITH SEPARATE TEXTURES/PLATEIA_DIMARCHON_FRAGMENT_-_18_07_2022.fbx")

print("Reading mesh...")
mesh = o3d.io.read_triangle_mesh(mesh_path, print_progress=True)

print("Creating trajectory...")
trajectory = create_trajectory(base_path)

print("Traversing trajectory...")
traverse_and_save_frames(mesh, trajectory, base_path, WIDTH, HEIGHT)

print("Done!...")
end = time.time()
elapsed_time = end - start
print("Time taken (s): " + str(elapsed_time))

exit()