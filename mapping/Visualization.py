#import pptk
import numpy as np
import open3d as o3d

def visualisePointCloud(pcd):
    print("to do visualization 21/07, should work... Figure out the cropping function to reduce data load!")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.run()


