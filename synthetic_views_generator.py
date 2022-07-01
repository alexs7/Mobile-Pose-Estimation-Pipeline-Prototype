import time
import numpy as np
import open3d as o3d
import os
import cv2
from scipy.spatial.transform import Rotation as R

DEG2RAD = np.pi / 180

def eulerToRotMRC(y, p, r): #radians
    cx = np.cos(r)
    cy = np.cos(p)
    cz = np.cos(y)
    sx = np.sin(r)
    sy = np.sin(p)
    sz = np.sin(y)
    return np.array([[ cx * cz + sx * sy * sz, -cx * sz + cz * sx * sy, -cy * sx],
                     [ -cy * sz, -cy * cz, -sy ],
                     [ cx * sy * sz - cz * sx, cx * cz * sy + sx * sz, -cx * cy ]])

def parseCamParam(cam):
    frame_name = cam[0]
    print("frame_name: " + frame_name)
    cam_center_cx = cam[1] #world
    cam_center_cy = cam[2]
    cam_center_cz = cam[3]
    # are in degrees
    y = cam[4] # heading / yaw, z
    p = cam[5] # pitch, y
    r = cam[6] # roll, x
    img_path = os.path.join(imgs_path, frame_name)
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    f = 935.3 #cam[7] #fx, fy
    px = 959.5 #cam[8]
    py = 539.5 #cam[9]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, f, f, px, py)
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = intrinsics
    # rotm = o3d.geometry.Geometry3D.get_rotation_matrix_from_zyx(rotation = [y * DEG2RAD, p * DEG2RAD, r * DEG2RAD])
    # rotm = R.from_euler('zyx', [y * DEG2RAD, p * DEG2RAD, r * DEG2RAD], degrees=False).as_dcm()
    rotm = eulerToRotMRC(y * DEG2RAD, p * DEG2RAD, r * DEG2RAD)
    # rotm = np.eye(3)
    trans = [cam_center_cx, cam_center_cy, cam_center_cz]
    extrinsics = np.r_[np.c_[rotm, trans], np.array([0, 0, 0, 1]).reshape(1,4)]
    cam_params.extrinsic = extrinsics
    return cam_params

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

print("Loading objects...")
cams_path = "/Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/1st MODEL - FILES/Internal_ExternalCameraParameters/Internal_external_1st_Model.csv"
mesh_path = "/Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/1st MODEL - FILES/EXPORT_Mesh/1st_MODEL_-_4k_Video_Photogrammetry.fbx"
imgs_path = "/Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/IMAGES/C0002 frames"
bundler_file_path_right_handed = "/Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/1st MODEL - FILES/BUNDLER/1st_MODEL_-_4k_Video_Photogrammetry.out"
bundler_file_path_left_handed = "/Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/Model 1 - Green Line Wall/1st MODEL - FILES/BUNDLER/Positive_1st_MODEL_-_4k_Video_Photogrammetry.out"

# switch file here
with open(bundler_file_path_right_handed) as f:
    bundler_file_right_handed = f.readlines()

cams_csv = np.loadtxt(cams_path, dtype='object, float, float, float, float, float, float, float, float, float, float, float, float, float', usecols=(range(0,14)), delimiter=',')
cams_bundler = create_cams_from_bundler(bundler_file_right_handed, cams_csv)

origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
print("Reading mesh...")
mesh = o3d.io.read_triangle_mesh(mesh_path)

points = getPoints(cams_csv)
colors = [[1, 0, 0] for i in range(len(points))]
pointcloud = o3d.geometry.PointCloud()
pointcloud.points = o3d.utility.Vector3dVector(points)
pointcloud.colors = o3d.utility.Vector3dVector(colors)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.add_geometry(origin)
vis.add_geometry(pointcloud)

#  second attempt bundler poses - right handed
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

    vis.add_geometry(cam_vis)
    vis.add_geometry(cam_vis_coor_sys)

    trajectory_cam = o3d.camera.PinholeCameraParameters()
    trajectory_cam.intrinsic = cam.intrinsic
    trajectory_cam.extrinsic = extrinsics
    trajectory_cams.append(trajectory_cam)

# first attempt
# for cam in cams_csv:
#     cam_params = parseCamParam(cam)
#     cam_center_cx = cam[1]  # world
#     cam_center_cy = cam[2]
#     cam_center_cz = cam[3]
#     trans = np.array([cam_center_cx, cam_center_cy, cam_center_cz])
#
#     # experimenting
#     # rot_fix = np.array([[0, -1, 0] , [-1, 0, 0] , [0, 0, -1]])
#     # rot_fix_1 = np.array([[-1, 0, 0] , [0, 1, 0] , [0, 0, -1]])
#     # rot_fix = np.matmul(rot_fix_0, rot_fix_1)
#     # rot_fix = np.array([[1, 1, 1] , [-1, -1, -1] , [-1, -1, -1]])
#     # rot_fix = np.array([[1, 0, 0] , [0, -1, 0] , [0, 0, -1]])
#     extrinsics = cam_params.extrinsic
#     rot_mat = extrinsics[0:3,0:3] # in camera coordinates
#     rot_mat = np.array([rot_mat[0,0:3] , -rot_mat[1,0:3] , -rot_mat[2,0:3]])
#     # rot_mat_fixed = np.matmul(rot_mat, rot_fix)
#     extrinsics = np.r_[np.c_[rot_mat, trans], np.array([0, 0, 0, 1]).reshape(1, 4)]
#     # extrinsics = np.r_[np.c_[rot_mat_fixed, trans], np.array([0, 0, 0, 1]).reshape(1,4)]
#     # extrinsics = np.r_[np.c_[rot_mat, trans], np.array([0, 0, 0, 1]).reshape(1,4)]
#     # extrinsics = np.r_[np.c_[np.eye(3), trans], np.array([0, 0, 0, 1]).reshape(1,4)]
#
#     # breakpoint()
#     cam_vis = o3d.geometry.LineSet.create_camera_visualization(cam_params.intrinsic.width, cam_params.intrinsic.height,
#                                                                cam_params.intrinsic.intrinsic_matrix, np.linalg.inv(extrinsics))
#
#     cam_vis_coor_sys = o3d.geometry.TriangleMesh.create_coordinate_frame(origin = trans)
#     cam_vis_coor_sys.rotate(np.eye(3))
#
#     vis.add_geometry(cam_vis)
#     vis.add_geometry(cam_vis_coor_sys)

local_params = o3d.io.read_pinhole_camera_parameters("/Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/camera_initial_position.json") #this was created manually
vis.get_view_control().convert_from_pinhole_camera_parameters(local_params, allow_arbitrary = False)


vis.get_view_control().convert_from_pinhole_camera_parameters(trajectory_cam, allow_arbitrary = False)
# o3d.io.write_pinhole_camera_parameters("/Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/camera_test_debug.json", test_param)

vis.run()  # user changes the view and press "q" to terminate

# param = vis.get_view_control().convert_to_pinhole_camera_parameters()
# o3d.io.write_pinhole_camera_parameters("/Users/alex/Projects/CYENS/fullpipeline_cyens/cyens_data/camera_test_new.json", param)

vis.destroy_window()

exit()

trajectory = o3d.camera.PinholeCameraTrajectory()
trajectory_list = []

for cam in cams_csv:
    frame_name = cam[0]
    print("frame_name: " + frame_name)
    cam_center_cx = cam[1] #world
    cam_center_cy = cam[2]
    cam_center_cz = cam[3]
    # are in degrees
    y = cam[4] # heading / yaw, z
    p = cam[5] # pitch, y
    r = cam[6] # roll, x
    img_path = os.path.join(imgs_path, frame_name)
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    f = cam[7] #fx, fy
    px = cam[8]
    py = cam[9]
    intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, f, f, px, py)
    cam_params = o3d.camera.PinholeCameraParameters()
    cam_params.intrinsic = intrinsics
    rotm = eulerToRotMRC(y * DEG2RAD, p * DEG2RAD, r * DEG2RAD)
    trans = [cam_center_cx, cam_center_cy, cam_center_cz]
    extrinsics = np.r_[np.c_[rotm, trans], np.array([0, 0, 0, 1]).reshape(1,4)]
    cam_params.extrinsic = extrinsics
    trajectory_list.append(cam_params)

trajectory.parameters = trajectory_list

def custom_draw_geometry_with_camera_trajectory(mesh, trajectory):
    custom_draw_geometry_with_camera_trajectory.index = 0
    custom_draw_geometry_with_camera_trajectory.trajectory = trajectory
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer()
    # if not os.path.exists("../../TestData/image/"):
    #     os.makedirs("../../TestData/image/")
    # if not os.path.exists("../../TestData/depth/"):
    #     os.makedirs("../../TestData/depth/")

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
            ctr.convert_from_pinhole_camera_parameters(glb.trajectory.parameters[glb.index], True)
            print("Capture image {:05d}".format(glb.index))
            time.sleep(2)
            # depth = vis.capture_depth_float_buffer(False)
            # image = vis.capture_screen_float_buffer(False)
            # plt.imsave("../../TestData/depth/{:05d}.png".format(glb.index),\
            #         np.asarray(depth), dpi = 1)
            # plt.imsave("../../TestData/image/{:05d}.png".format(glb.index),\
            #         np.asarray(image), dpi = 1)
            # vis.capture_depth_image("image_{:05d}.png".format(glb.index), False)
            # vis.capture_screen_image("depth_{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1

        if glb.index == len(glb.trajectory.parameters):
            print("Done!")
            custom_draw_geometry_with_camera_trajectory.vis.register_animation_callback(None)

        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(mesh)
    # vis.get_render_option().load_from_json("../../TestData/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()

custom_draw_geometry_with_camera_trajectory(mesh, trajectory)

# previous code to use in need of references

# cameras = o3d.io.read_triangle_mesh("/Users/alex/Projects/CYENS/andreas_models/NEWER VERSION - 30 05 2022 - with camera objects/cameras_only/FACADE ALIGNMENT with camera objects.fbx")

# o3d.visualization.draw_geometries([cameras, origin], width=1000, height=800, mesh_show_wireframe=False)

# vis = o3d.visualization.VisualizerWithVertexSelection()

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(mesh)
# vis.run()  # user changes the view and press "q" to terminate
# param = vis.get_view_control().convert_to_pinhole_camera_parameters()
# o3d.io.write_pinhole_camera_parameters("/Users/alex/Projects/CYENS/andreas_models/NEWER VERSION - 30 05 2022 - with camera objects/camera_1.json", param)
# vis.destroy_window()

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# ctr = vis.get_view_control()
# param = o3d.io.read_pinhole_camera_parameters("/Users/alex/Projects/CYENS/andreas_models/NEWER VERSION - 30 05 2022 - with camera objects/camera.json")
# vis.add_geometry(mesh)
# vis.add_geometry(origin)
# ctr.convert_from_pinhole_camera_parameters(param)
# vis.run()
# vis.destroy_window()

