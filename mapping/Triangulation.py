import os
import cv2
import numpy as np
import cv2 as cv
import open3d as o3d
from .InputParameters import *
import csv
from .Visualization import *

## Function to print the used parameters for calibration
def PrintParameters():
    import pickle
    os.chdir(WorkingDirectory)
    with open('RotationMatrix.pkl', 'rb') as file3:
        rot = pickle.load(file3)
    with open('TranslationMatrix.pkl', 'rb') as file4:
        trans = pickle.load(file4)
    with open('ProjectionMatrixLeftCam.pkl', 'rb') as file1:
        p1 = pickle.load(file1)
    with open('ProjectionMatrixRightCam.pkl', 'rb') as file2:
        p2 = pickle.load(file2)

    with open('RotatieMatrixRechterCamera.csv', 'w+', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=' ')
        my_writer.writerow(rot)
    with open('TranslatieMatrixRechterCamera.csv', 'w+', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=' ')
        my_writer.writerow(trans)



## Function for finding XYZ Coordinates of point, These XYZ coordinates are saved in a Point Cloud.ply file in the main directory
def Triangulate():
    import Detection
    import pickle

    testarray = []
    colors = []
    i = 0


    with open('ProjectionMatrixLeftCam.pkl', 'rb') as file1:
        p1 = pickle.load(file1)
    with open('ProjectionMatrixRightCam.pkl', 'rb') as file2:
        p2 = pickle.load(file2)
    with open('CameraMatrixL.pkl', 'rb') as file3:
        IntrinsicMatrixL = pickle.load(file3)
    with open('CameraMatrixR.pkl','rb') as file4:
        IntrinsicMatrixR = pickle.load(file4)
    with open ('DistortionL.pkl','rb') as file5:
        DistL= pickle.load(file5)
    with open ('DistortionR.pkl','rb') as file6:
        DistR = pickle.load(file6)

    with open('IntrinsiekeMatrixRechterCamera.csv', 'w+', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=' ')
        my_writer.writerow(IntrinsicMatrixR)
    with open('IntrinsiekeMatrixLinkerCamera.csv', 'w+', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=' ')
        my_writer.writerow(IntrinsicMatrixL)


    ## Reading image for colors of point cloud
    os.chdir(InputParameters.CalibratedImageDirectory)
    img = cv.imread('imgHorCAML9.png', cv.IMREAD_UNCHANGED)
    os.chdir(InputParameters.WorkingDirectory)


    inversLeftCam,inversRightCam = Detection.FindCorrespondence()
    ### Finding shape for looping
    shapeInversLeft = np.shape(inversLeftCam)
    shapeInversRight = np.shape(inversRightCam)

    methodOfTriangulation = InputParameters.methodOfTriangulation

    if methodOfTriangulation == 1:
        for ii in range(0, min(shapeInversLeft[0], shapeInversRight[0]) - 1):
            for jj in range(0, min(shapeInversLeft[1], shapeInversRight[1]) - 1):
                if inversLeftCam[ii][jj][0] != 0 and inversLeftCam[ii][jj][1] != 0 and inversRightCam[ii][jj][
                    0] != 0 and inversRightCam[ii][jj][1] != 0:
                    projpoints1 = np.array([[inversLeftCam[ii][jj][0], inversLeftCam[ii][jj][1]],
                                            [inversRightCam[ii][jj][0], inversRightCam[ii][jj][1]]], dtype=np.float32)

                    points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                    points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                    testarray.append(points3D[:3])

                    ## Array for colors
                    a = inversLeftCam[ii][jj][0]
                    b = inversLeftCam[ii][jj][1]
                    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                    colors.append(img[int(b)][int(a)] / 255)

    elif methodOfTriangulation == 2:
        for ii in range(0,min(shapeInversLeft[0],shapeInversRight[0])-1):
            for jj in range(0,min(shapeInversLeft[1],shapeInversRight[1])-1):
                flag = False
                if inversLeftCam[ii][jj][0] != 0 and inversLeftCam[ii][jj][1] != 0 and inversRightCam[ii][jj][0] != 0 and inversRightCam[ii][jj][1] != 0:

                    projpoints1 = np.array([[inversLeftCam[ii][jj][0],inversLeftCam[ii][jj][1]],
                                            [inversRightCam[ii][jj][0], inversRightCam[ii][jj][1]]], dtype=np.float32)

                    points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                    points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                    testarray.append(points3D[:3])
                    a = inversLeftCam[ii][jj][0]
                    b = inversLeftCam[ii][jj][1]
                    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                    colors.append(img[int(b)][int(a)] / 255)
                elif inversLeftCam[ii][jj][0] != 0 and inversLeftCam[ii][jj][1] != 0 and inversRightCam[ii][jj][0] == 0 and inversRightCam[ii][jj][1] == 0:
                    if inversRightCam[ii][jj+1][0] != 0 and inversRightCam[ii][jj+1][1] != 0:
                        projpoints1 = np.array([[inversLeftCam[ii][jj][0], inversLeftCam[ii][jj][1]],
                                            [inversRightCam[ii][jj+1][0], inversRightCam[ii][jj+1][1]]], dtype=np.float32)
                        points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                        points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                        testarray.append(points3D[:3])
                        flag = True
                        a = inversLeftCam[ii][jj][0]
                        b = inversLeftCam[ii][jj][1]
                        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                        colors.append(img[int(b)][int(a)] / 255)
                    if inversRightCam[ii][jj-1][0] != 0 and inversRightCam[ii][jj-1][1] != 0 and flag == False:
                        projpoints1 = np.array([[inversLeftCam[ii][jj][0], inversLeftCam[ii][jj][1]],
                                                [inversRightCam[ii][jj-1][0], inversRightCam[ii][jj-1][1]]],
                                               dtype=np.float32)
                        points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                        points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                        testarray.append(points3D[:3])
                        flag = True
                        a = inversLeftCam[ii][jj][0]
                        b = inversLeftCam[ii][jj][1]
                        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                        colors.append(img[int(b)][int(a)] / 255)
                    if inversRightCam[ii+1][jj][0] != 0 and inversRightCam[ii+1][jj][1] != 0 and flag == False:
                        projpoints1 = np.array([[inversLeftCam[ii][jj][0], inversLeftCam[ii][jj][1]],
                                                [inversRightCam[ii+1][jj][0], inversRightCam[ii+1][jj][1]]],
                                               dtype=np.float32)
                        points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                        points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                        testarray.append(points3D[:3])
                        flag = True
                        a = inversLeftCam[ii][jj][0]
                        b = inversLeftCam[ii][jj][1]
                        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                        colors.append(img[int(b)][int(a)] / 255)
                    if inversRightCam[ii-1][jj][0] != 0 and inversRightCam[ii-1][jj][1] != 0 and flag == False:
                        projpoints1 = np.array([[inversLeftCam[ii][jj][0], inversLeftCam[ii][jj][1]],
                                                [inversRightCam[ii-1][jj][0], inversRightCam[ii-1][jj][1]]],
                                               dtype=np.float32)
                        points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                        points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                        testarray.append(points3D[:3])
                        flag = True
                        a = inversLeftCam[ii][jj][0]
                        b = inversLeftCam[ii][jj][1]
                        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                        colors.append(img[int(b)][int(a)] / 255)
                elif inversLeftCam[ii][jj][0] == 0 and inversLeftCam[ii][jj][1] == 0 and inversRightCam[ii][jj][0] != 0 and inversRightCam[ii][jj][1] != 0:
                    if inversLeftCam[ii][jj + 1][0] != 0 and inversLeftCam[ii][jj + 1][1] != 0:
                        projpoints1 = np.array([[inversLeftCam[ii][jj + 1][0], inversLeftCam[ii][jj + 1][1]],
                                                [inversRightCam[ii][jj][0], inversRightCam[ii][jj + 1][1]]],
                                               dtype=np.float32)
                        points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                        points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                        testarray.append(points3D[:3])
                        flag = True
                        a = inversLeftCam[ii][jj+1][0]
                        b = inversLeftCam[ii][jj+1][1]
                        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                        colors.append(img[int(b)][int(a)] / 255)
                    if inversLeftCam[ii][jj - 1][0] != 0 and inversLeftCam[ii][jj - 1][1] != 0 and flag == False:
                        projpoints1 = np.array([[inversLeftCam[ii][jj - 1][0], inversLeftCam[ii][jj - 1][1]],
                                                [inversRightCam[ii][jj][0], inversRightCam[ii][jj][1]]],
                                               dtype=np.float32)
                        points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                        points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                        testarray.append(points3D[:3])
                        flag = True
                        a = inversLeftCam[ii][jj-1][0]
                        b = inversLeftCam[ii][jj-1][1]
                        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                        colors.append(img[int(b)][int(a)] / 255)
                    if inversLeftCam[ii + 1][jj][0] != 0 and inversLeftCam[ii + 1][jj][1] != 0 and flag == False:
                        projpoints1 = np.array([[inversLeftCam[ii + 1][jj][0], inversLeftCam[ii + 1][jj][1]],
                                                [inversRightCam[ii][jj][0], inversRightCam[ii][jj][1]]],
                                               dtype=np.float32)
                        points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                        points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                        testarray.append(points3D[:3])
                        flag = True
                        a = inversLeftCam[ii+1][jj][0]
                        b = inversLeftCam[ii+1][jj][1]
                        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                        colors.append(img[int(b)][int(a)] / 255)
                    if inversLeftCam[ii - 1][jj][0] != 0 and inversLeftCam[ii - 1][jj][1] != 0 and flag == False:
                        projpoints1 = np.array([[inversLeftCam[ii - 1][jj][0], inversLeftCam[ii - 1][jj][1]],
                                                [inversRightCam[ii][jj][0], inversRightCam[ii][jj][1]]],
                                               dtype=np.float32)
                        points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                        points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                        testarray.append(points3D[:3])
                        flag = True
                        a = inversLeftCam[ii-1][jj][0]
                        b = inversLeftCam[ii-1][jj][1]
                        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                        colors.append(img[int(b)][int(a)] / 255)



                ##Putting points in CSV file
                #with open('CorrespondentiePunten.csv', 'a+', newline='') as csvfile:
                    #my_writer = csv.writer(csvfile)
                    #my_writer.writerow(projpoints1)


    testarray = np.array(testarray)         ## XYZ Points
    colors = np.array(colors)               ## Matching RGB values


    ## Writing PLY file trhough open3D
    xyz = np.array(testarray)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)                ## Put Nx3 array in o3d format
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud("./Point Cloud.ply", pcd, write_ascii=True)


    ######################################VISUALIZATION#################################################################
    Visualization.visualisePointCloud(testarray,colors)


## Currently Threshold picture is being used instead of taking a separate picture for colors
def TakeColorImage():
    import Camera1
    image = Camera1.Webcam()
    image.OpenCAMR(1)
    ret, imageframe = image.vid.read()
    colors = cv.cvtColor(imageframe, cv.COLOR_BGR2RGB)
    return colors

## Not being used currently
def CreatePLY(vertices,colors,filename):

    vertices = np.hstack([vertices.reshape(-1,3),colors])
    ply_header = '''ply 
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename,'w') as f:
        f.write(ply_header %dict(vert_num = len(vertices)))
        np.savetxt(f,vertices, '%f %f %f %d %d %d')




