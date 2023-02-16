import glob
import cv2 as cv
import numpy as np
import Graycode
import os
import Camera1
import pickle
import InputParameters
import csv

chessboardSize = (12,7)
frameSize = (640,480)

LeftCamera = InputParameters.LeftCamera
RightCamera = InputParameters.RightCamera


camL = Camera1.Webcam()
camR = Camera1.Webcam()
camL.OpenCAML(LeftCamera)
camR.OpenCAMR(RightCamera)
counter = 0
print('Please take a minimum of 10 Images.\nPress S to take image of calibration pattern.\nPress Q for 3 sec if enough images are taken.\nIf new calibration is unnecessary, press Q and previous calibration parameters will be used')
while True :
    try:
        os.chdir(InputParameters.ChessboardImagesDirectory)
        ret, frameL = camL.vid.read()
        cv.imshow('Left Camera',frameL)
        ret, frameR = camR.vid2.read()
        cv.imshow('Right Camera',frameR)

        if cv.waitKey(1) == ord('s'):
                cv.imwrite("ChessboardLeftCam{}.png".format(counter), frameL)
                cv.imwrite("ChessboardRightCam{}.png".format(counter), frameR)
                counter = counter + 1
                print(counter, 'Pictures Taken')
        elif cv.waitKey(1) == ord('q'):
                break
    except:
        print("no cameras found, using previous images to determine parameters...")
        break
## Termination Criteria for iterative subpixel loop
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

## Prepare object points like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((12 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:12, 0:7].T.reshape(-1, 2)           ##https://answers.opencv.org/question/25022/stereo-calibration-baseline-in-meters/ NAKIJKEN!!!!
objp = objp * 19                                            ## The distance between squares is 25mm, this way the translation vector is in mm unit


## Arrays to store object points and image points from all the images.
objPoints = []  # 3d point in real world space
imgPointsL = []  # 2d points in image plane.
imgPointsR = []


imagesLeft = glob.glob('ChessboardLeftCam*.png')
imagesRight = glob.glob('ChessboardRightCam*.png')           # Hier moeten nog de rechtse foto's komen

for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:
        objPoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        imgPointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        imgPointsR.append(cornersR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(100)
    else:
        print("Error in finding chessboard corners")

cv.destroyAllWindows()

retL, cameraMatrixL, distL, rvecsL, tvecsL= cv.calibrateCamera(objPoints, imgPointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1,
                                                               (widthL, heightL))
print("RMS error Left Camera :",retL)   ##Error should be between 0 and 1 (as close to 0 as possible)

retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objPoints, imgPointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1,
                                                               (widthR, heightR))
print("RMS Error Right Camera :",retR)  ##Error should be between 0 and 1 (as close to 0 as possible)
print ('Intrinsic Matrix Left Camera : ',cameraMatrixL)
print('Intrinsic Matrix Right Camera : ',cameraMatrixR)

## Save camera properties for possible later use
os.chdir(InputParameters.WorkingDirectory)
with open('CameraMatrixL.pkl', 'wb') as file1:
    pickle.dump(cameraMatrixL, file1)
with open('CameraMatrixR.pkl', 'wb') as file2:
    pickle.dump(cameraMatrixR, file2)
with open('DistortionL.pkl', 'wb') as file3:
    pickle.dump(distL, file3)
with open('DistortionR.pkl', 'wb') as file4:
    pickle.dump(distR, file4)



################################### CALIBRATION BETWEEN 2 CAMERAS ############################################################
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(
    objPoints, imgPointsL, imgPointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1],
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


with open('IntrinsiekeMatrixLinkerCamera.csv', 'w+', newline='') as csvfile:
    csvfile.write(IntrinsicMatrixL)
with open('IntrinsiekeMatrixRechterCamera.csv', 'w+', newline='') as csvfil:
    csvfil.write(IntrinsicMatrixR)
with open('ExtrinsiekeMatrixRechterCamera.csv', 'w+', newline='') as csvfile:
    my_writer = csv.writer(csvfile, delimiter=' ')
    my_writer.writerow(np.concatenate([rot, trans], axis=-1))












