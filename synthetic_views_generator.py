import time

import open3d as o3d

print("Loading model...")

origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh = o3d.io.read_triangle_mesh("/Users/alex/Projects/CYENS/andreas_models/NEWER VERSION - 30 05 2022 - with camera objects/model_only/30_05_2022_-ONE_FACADE_-_ALEX_REPROJECTION_PROJECT_-_RC_FILE_Model_5.fbx")

trajectory = o3d.camera.PinholeCameraTrajectory()
trajectory_list = []

cam_0 = o3d.io.read_pinhole_camera_parameters("/Users/alex/Projects/CYENS/andreas_models/NEWER VERSION - 30 05 2022 - with camera objects/camera_0.json")
cam_1 = o3d.io.read_pinhole_camera_parameters("/Users/alex/Projects/CYENS/andreas_models/NEWER VERSION - 30 05 2022 - with camera objects/camera_1.json")
cam_2 = o3d.io.read_pinhole_camera_parameters("/Users/alex/Projects/CYENS/andreas_models/NEWER VERSION - 30 05 2022 - with camera objects/camera_2.json")

trajectory_list.append(cam_0)
trajectory_list.append(cam_1)
trajectory_list.append(cam_2)

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
            time.sleep(0.1)
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

