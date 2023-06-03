#import pptk
import numpy as np
import open3d as o3d

def visualisePointCloud(pcd):
    print("to do visualization 21/07, should work... Figure out the cropping function to reduce data load!")
    vis = o3d.visualization.Visualizer()
    cl,idx = pcd.remove_radius_outlier(nb_points=16,radius=0.1)
    #pcd = pcd.select_by_index(idx)
    print(len(np.asarray(pcd.points)))
    vis.create_window()
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.run()


