import glob
import cv2 as cv
import numpy as np
import mapping.Graycode as Graycode
import os
import mapping.Image_processor as Image_processor
import pickle
import mapping.InputParameters as InputParameters
import csv
import copy
import matplotlib.pyplot as plt
from PyCBD.pipelines import CBDPipeline, prepare_image
from PyGeiger.detector import GeigerDetector
from PyCBD.checkerboard_detection.detectors import CheckerboardDetector

chessboardSize = (9,6)
frameSize = (640,480)

LeftCamera = InputParameters.LeftCamera
RightCamera = InputParameters.RightCamera
detector = CBDPipeline(detector=GeigerDetector() ,expand=True, predict=True)
detector.must_plot_iterations = False

cams = Image_processor.Image_Handle(test = True, location_L =LeftCamera, location_R =RightCamera)
cams.OpenCAM()
## Termination Criteria for iterative subpixel loop
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

## Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)           ##https://answers.opencv.org/question/25022/stereo-calibration-baseline-in-meters/ NAKIJKEN!!!!
objp = objp * 19                                            ## The distance between squares is 25mm, this way the translation vector is in mm unit

## Arrays to store object points and image points from all the images.
objPoints = []  # 3d point in real world space
imgPointsL = []  # 2d points in image plane.
imgPointsR = []

def picture_taking():
    counter = 0
    print('Please take a minimum of 10 Images\n')
    while True :
        os.chdir(InputParameters.ChessboardImagesDirectory)
        frameL, frameR = cams.GetCalibrateFrame()
        plt.imshow(frameL)
        plt.show()
        plt.imshow(frameR)
        plt.show()
        reply = input("Reply with s if checkerboard is fully in image, press enter to simply run again, press q to quit.\n")
        if "s" == reply:
                imgL = copy.deepcopy(frameL)
                imgR = copy.deepcopy(frameR)
                grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
                grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
                checkerboard_drawing(imgL,imgR,grayL,grayR)
                affirm = input("type y if happy:\n")
                if affirm == "y":
                    cv.imwrite("ChessboardLeftCam{}.tiff".format(counter), cv.cvtColor(frameL, cv.COLOR_RGB2BGR))
                    cv.imwrite("ChessboardRightCam{}.tiff".format(counter), cv.cvtColor(frameR, cv.COLOR_RGB2BGR))
                    counter = counter + 1
                    print(counter, 'Pictures Saved')
                    newmatrixL, matrixL, newmatrixR, matrixR, dL, dR = matrix_determination(imgL,imgR)
        elif "q" == reply:
            break
    newmatrixL, matrixL, newmatrixR, matrixR, dL, dR = matrix_determination(imgL,imgR)
    save_load(newmatrixL,matrixL,newmatrixR,matrixR,grayL,dL,dR)               

def checkerboard_drawing(rgbL,rgbR,left,right):
    # Find the chess board corners
    results_L, board_uv_L, board_xy_L = detector.detect_checkerboard(rgbL, chessboardSize)
    results_R, board_uv_R, board_xy_R = detector.detect_checkerboard(rgbR, chessboardSize)
    #board_xy_L = np.hstack((board_xy_L,np.zeros((len(board_xy_L),1))))
    #board_xy_R = np.hstack((board_xy_R,np.zeros((len(board_xy_R),1))))
    print(len(board_uv_L) == len(objp))
    print(len(board_uv_R) == len(objp))
    if results_L and results_R == 1 or 2 :
        objPoints.append(objp)

        #cornersL = cv.cornerSubPix(left, board_uv_L, (11, 11), (-1, -1), criteria)
        imgPointsL.append(np.array(board_uv_L).astype(np.float32))

        #cornersR = cv.cornerSubPix(right, board_uv_R, (11, 11), (-1, -1), criteria)
        imgPointsR.append(np.array(board_uv_R).astype(np.float32))
        #cv.drawChessboardCorners(rgbL, chessboardSize, cornersL)
        # Draw and display the corner
        plt.imshow(rgbL)
        plt.plot(board_uv_L[:, 0], board_uv_L[:, 1], 'go', markeredgecolor='k')
        for i in range(0, board_uv_L.shape[0]):
            plt.text(board_uv_L[i, 0], board_uv_L[i, 1], i, color="black")
        title = "Detected checkerboard Left"
        plt.title(title)
        plt.show()

        plt.imshow(rgbR)
        plt.plot(board_uv_R[:, 0], board_uv_R[:, 1], 'go', markeredgecolor='k')
        for i in range(0, board_uv_R.shape[0]):
            plt.text(board_uv_R[i, 0], board_uv_R[i, 1], i, color="black")
        title = "Detected checkerboard Right"
        plt.title(title)
        plt.show()
    else:
        print("Error in finding chessboard corners")

def matrix_determination(left,right):

    retL, cameraMatrixL, distL, rvecsL, tvecsL= cv.calibrateCamera(objPoints,imgPointsL, frameSize, None, None)
    heightL, widthL, channelsL = left.shape
    newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1,
                                                                (widthL, heightL))
    print("RMS error Left Camera :",retL)   ##Error should be between 0 and 1 (as close to 0 as possible)

    retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objPoints, imgPointsR, frameSize, None, None)
    heightR, widthR, channelsR = right.shape
    newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1,
                                                                (widthR, heightR))
    print("RMS Error Right Camera :",retR)  ##Error should be between 0 and 1 (as close to 0 as possible)
    print ('Intrinsic Matrix Left Camera : ',cameraMatrixL)
    print('Intrinsic Matrix Right Camera : ',cameraMatrixR)
    return [newCameraMatrixL, cameraMatrixL, newCameraMatrixR, cameraMatrixR, distL, distR]

def save_load(newCameraMatrixL,cameraMatrixL,newCameraMatrixR,cameraMatrixR,gray,distorL,distorR):
    distL = distorL
    distR = distorR
    ## Save camera properties for possible later use
    os.chdir(InputParameters.WorkingDirectory)
    with open('CameraMatrixL.pkl', 'wb') as file1:
        pickle.dump(cameraMatrixL, file1)
    with open('CameraMatrixR.pkl', 'wb') as file2:
        pickle.dump(cameraMatrixR, file2)
    with open('DistortionL.pkl', 'wb') as file3:
        pickle.dump(distorL, file3)
    with open('DistortionR.pkl', 'wb') as file4:
        pickle.dump(distorR, file4)


    ################################### CALIBRATION BETWEEN 2 CAMERAS ############################################################
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
        objPoints, imgPointsL, imgPointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, gray.shape[::-1],
        criteria_stereo, flags)


    #################################### PROJECTION MATRICES #############################################################################
    # Projection matrices for both cameras (P = M*R|t)     (left camera uses the identity matrix)

    p1 = cameraMatrixL @ np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    p2 = cameraMatrixR @ np.concatenate([rot, trans], axis=-1)


    # Save calibration parameters in file
    os.chdir(InputParameters.WorkingDirectory)
    with open('ProjectionMatrixLeftCam.pkl', 'wb') as file1:
        pickle.dump(p1, file1)
    with open('ProjectionMatrixRightCam.pkl', 'wb') as file2:
        pickle.dump(p2, file2)
    with open('RotationMatrix.pkl', 'wb') as file3:
        pickle.dump(rot, file3)
    with open('TranslationMatrix.pkl', 'wb') as file4:
        pickle.dump(trans, file4)


    os.chdir(InputParameters.WorkingDirectory)
    with open('CameraMatrixL.pkl', 'rb') as file3:
        IntrinsicMatrixL = pickle.load(file3)
    with open('CameraMatrixR.pkl', 'rb') as file4:
        IntrinsicMatrixR = pickle.load(file4)


    with open('IntrinsicMatrixLCam.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(IntrinsicMatrixL)
    with open('IntrinsicMatrixRCam.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(IntrinsicMatrixR)
    with open('ExtrinsicMatrixRCam.csv', 'w', newline='') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=' ')
        my_writer.writerow(np.concatenate([rot, trans], axis=-1))

def load_existing_img():
    os.chdir(InputParameters.ChessboardImagesDirectory)
    imagesLeft = glob.glob('ChessboardLeftCam*.tiff')
    imagesRight = glob.glob('ChessboardRightCam*.tiff')           # Hier moeten nog de rechtse foto's komen
    for imgLeft, imgRight in zip(imagesLeft, imagesRight):
        imgL = cv.imread(imgLeft)
        imgR = cv.imread(imgRight)
        grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
        grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
        checkerboard_drawing(imgL,imgR,grayL,grayR)
    newmatrixL, matrixL, newmatrixR, matrixR, dL, dR = matrix_determination(left = imgL,right = imgR)
    return [newmatrixL, matrixL, newmatrixR, matrixR, dL, dR, grayL]

if __name__ == "__main__":
    calibrate = False
    if calibrate:    
        picture_taking()
    else:
        newmatrixL, matrixL, newmatrixR, matrixR, dL, dR, grayL = load_existing_img()
    save_load(newCameraMatrixL=newmatrixL, newCameraMatrixR=newmatrixR, cameraMatrixL=matrixL, cameraMatrixR=matrixR, gray=grayL, distorL=dL, distorR= dR)