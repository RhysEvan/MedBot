import os
import cv2
import numpy as np
import cv2 as cv
import open3d as o3d
import mapping.InputParameters as InputParameters
import csv
import mapping.Visualization as Visualization
import pickle
from mapping.Detection import Detecting

class triangulator():
    def __init__(self, parent):
        self.parent = parent
            
    ## Function to print the used parameters for calibration
    def PrintParameters(self):
        os.chdir(InputParameters.WorkingDirectory)
        with open('RotationMatrix.pkl', 'rb') as file3:
            rot = pickle.load(file3)
            print('Rotation Matrix : ',rot)
        with open('TranslationMatrix.pkl', 'rb') as file4:
            trans = pickle.load(file4)
            print('Translation Matrix : ', trans)
        with open('ProjectionMatrixLeftCam.pkl', 'rb') as file1:
            p1 = pickle.load(file1)
            print('Projection Matrix for Left Cam: ',p1)
        with open('ProjectionMatrixRightCam.pkl', 'rb') as file2:
            p2 = pickle.load(file2)
            print('Projection Matrix for Right Cam: ',p2)
        with open('RotationMatrixRCamera.csv', 'w+', newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter=' ')
            my_writer.writerow(rot)
        with open('TranslationMatrixRCamera.csv', 'w+', newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter=' ')
            my_writer.writerow(trans)

    ## Function for finding XYZ Coordinates of point, These XYZ coordinates are saved in a Point Cloud.ply file in the main directory
    def Triangulate(self):
        self.testarray = []
        self.colors = []
        i = 0
        with open('ProjectionMatrixLeftCam.pkl', 'rb') as file1:
            self.p1 = pickle.load(file1)
        with open('ProjectionMatrixRightCam.pkl', 'rb') as file2:
            self.p2 = pickle.load(file2)
        with open('CameraMatrixL.pkl', 'rb') as file3:
            IntrinsicMatrixL = pickle.load(file3)
        with open('CameraMatrixR.pkl','rb') as file4:
            IntrinsicMatrixR = pickle.load(file4)
        with open ('DistortionL.pkl','rb') as file5:
            DistL= pickle.load(file5)
        with open ('DistortionR.pkl','rb') as file6:
            DistR = pickle.load(file6)

        with open('IntrinsicMatrixRCamera.csv', 'w+', newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter=' ')
            my_writer.writerow(IntrinsicMatrixR)
        with open('IntrinsicMatrixLCamera.csv', 'w+', newline='') as csvfile:
            my_writer = csv.writer(csvfile, delimiter=' ')
            my_writer.writerow(IntrinsicMatrixL)


        ## Reading image for colors of point cloud
        img = self.parent.Horz_list[-1][0]

        inverseLeftCam,inverseRightCam = self.parent.detect.FindCorrespondence()
        ### Finding shape for looping
        shapeInversLeft = np.shape(inverseLeftCam)
        shapeInversRight = np.shape(inverseRightCam)

        methodOfTriangulation = InputParameters.methodOfTriangulation

        if methodOfTriangulation == 1:
            for ii in range(0, min(shapeInversLeft[0], shapeInversRight[0]) - 1):
                for jj in range(0, min(shapeInversLeft[1], shapeInversRight[1]) - 1):
                    if inverseLeftCam[ii][jj][0] != 0 and inverseLeftCam[ii][jj][1] != 0 and inverseRightCam[ii][jj][
                        0] != 0 and inverseRightCam[ii][jj][1] != 0:
                        projpoints1 = np.array([[inverseLeftCam[ii][jj][0], inverseLeftCam[ii][jj][1]],
                                                [inverseRightCam[ii][jj][0], inverseRightCam[ii][jj][1]]], dtype=np.float32)

                        points4D = cv.triangulatePoints(p1, p2, projpoints1[0], projpoints1[1])  # De correcte manier
                        points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
                        testarray.append(points3D[:3])

                        ## Array for colors
                        a = inverseLeftCam[ii][jj][0]
                        b = inverseLeftCam[ii][jj][1]
                        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                        colors.append(img[int(b)][int(a)] / 255)

        elif methodOfTriangulation == 2:
            for ii in range(0,min(shapeInversLeft[0],shapeInversRight[0])-1):
                for jj in range(0,min(shapeInversLeft[1],shapeInversRight[1])-1):
                    flag = False

                    leftframe = inverseLeftCam[ii][jj]
                    rightframe = inverseRightCam[ii][jj]
                    
                    if leftframe[0] != 0 and leftframe[1] != 0 and rightframe[0] != 0 and rightframe[1] != 0:
                        self.pointer(leftframe,rightframe)
                        self.array_colorizer(img, leftframe)

                    elif leftframe[0] != 0 and leftframe[1] != 0 and rightframe[0] == 0 and rightframe[1] == 0:

                        rightframe = inverseRightCam[ii][jj+1]

                        if rightframe[0] != 0 and rightframe[1] != 0:
                            self.pointer(leftframe,rightframe)
                            flag = True
                            self.array_colorizer(img, leftframe)
                        
                        rightframe = inverseRightCam[ii][jj-1]

                        if rightframe[0] != 0 and rightframe[1] != 0 and flag == False:
                            self.pointer(leftframe,rightframe)
                            flag = True
                            self.array_colorizer(img, leftframe)
                        
                        rightframe = inverseRightCam[ii+1][jj]
                        
                        if rightframe[0] != 0 and rightframe[1] != 0 and flag == False:
                            self.pointer(leftframe,rightframe)
                            flag = True
                            self.array_colorizer(img, leftframe)

                        rightframe = inverseRightCam[ii-1][jj]

                        if rightframe[0] != 0 and rightframe[1] != 0 and flag == False:
                            self.pointer(leftframe,rightframe)
                            flag = True
                            self.array_colorizer(img, leftframe)

                    elif leftframe[0] == 0 and leftframe[1] == 0 and rightframe[0] != 0 and rightframe[1] != 0:

                        leftframe = inverseLeftCam[ii][jj+1]

                        if leftframe[0] != 0 and leftframe[1] != 0:
                            self.pointer(leftframe, rightframe)
                            flag = True
                            self.array_colorizer(img, leftframe)
                        
                        leftframe = inverseLeftCam[ii][jj-1]

                        if leftframe[0] != 0 and leftframe[1] != 0 and flag == False:
                            self.pointer(leftframe,rightframe)
                            flag = True
                            self.array_colorizer(img, leftframe)

                        leftframe = inverseLeftCam[ii + 1][jj]

                        if leftframe[0] != 0 and leftframe[1] != 0 and flag == False:
                            self.pointer(leftframe,rightframe)
                            flag = True
                            self.array_colorizer(img, leftframe)
                        
                        leftframe = inverseLeftCam[ii - 1][jj]

                        if leftframe[0] != 0 and leftframe[1] != 0 and flag == False:
                            self.pointer(leftframe,rightframe)
                            flag = True
                            self.array_colorizer(img, leftframe)

        testarray = np.array(self.testarray)         ## XYZ Points
        colors = np.array(self.colors)               ## Matching RGB values
        ## Writing PLY file trhough open3D
        print("Number of points : ",np.shape(testarray))
        print("Number of colors :",np.shape(colors))
        xyz = np.array(testarray)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)                ## Put Nx3 array in o3d format
        pcd.colors = o3d.utility.Vector3dVector(colors)
        ######################################VISUALIZATION#################################################################
        Visualization.visualisePointCloud(pcd)

    def pointer(self, leftframe, rightframe):
        projpoints1 = np.array([[leftframe[0],leftframe[1]],
                                [rightframe[0], rightframe[1]]], dtype=np.float32)

        points4D = cv.triangulatePoints(self.p1, self.p2, projpoints1[0], projpoints1[1])  # De correcte manier
        points3D = points4D[:3] / points4D[3]  ## From Homogenous coordinates to Cartesian
        self.testarray.append(points3D[:3])

    def array_colorizer(self, img, frame):
        a = frame[0]
        b = frame[1]
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        self.colors.append(img[int(b)][int(a)] / 255)

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




