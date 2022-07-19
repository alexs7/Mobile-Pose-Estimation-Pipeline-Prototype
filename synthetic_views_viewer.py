import sys
import open3d as o3d
import os

mesh_path = os.path.join(sys.argv[1])
mesh = o3d.io.read_triangle_mesh(mesh_path)

vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1080)
vis.add_geometry(mesh)
vis.run()
vis.destroy_window()

