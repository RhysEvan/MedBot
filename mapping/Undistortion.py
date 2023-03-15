import pickle
import InputParameters
import os
import cv2 as cv
import Graycode


###################################################### UNDISTORTION OF TAKEN IMAGES ##################################################################
## Opening saved parameters
os.chdir(InputParameters.WorkingDirectory)
with open('CameraMatrixL.pkl', 'rb') as file1:
    cameraMatrixL = pickle.load(file1)
with open('CameraMatrixR.pkl', 'rb') as file2:
    cameraMatrixR = pickle.load(file2)
with open('DistortionL.pkl', 'rb') as file3:
    distL = pickle.load(file3)
with open('DistortionR.pkl', 'rb') as file4:
    distR = pickle.load(file4)


## Function for Undistorting Images
def Undistort(imgName, imgNameUndistorted, cameraMatrix, dist):
    os.chdir(InputParameters.ImageDirectory)
    img = cv.imread(imgName)
    h, w = img.shape[
               :2]  ## img.shape is function that returns the height & width of image(array) and color if it is a colorimage
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

    dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    os.chdir(InputParameters.CalibratedImageDirectory)
    cv.imwrite(imgNameUndistorted, dst)
    return cameraMatrix, dist


## Undistorting 40 Images through loop + 40 invers images
for i in range(Graycode.length):
    Undistort('imgVertCAML{}.png'.format(i), 'imgVertCAML{}.png'.format(i), cameraMatrixL, distL)
    Undistort('imgHorCAML{}.png'.format(i), 'imgHorCAML{}.png'.format(i), cameraMatrixL, distL)
    Undistort('imgVertCAMR{}.png'.format(i), 'imgVertCAMR{}.png'.format(i), cameraMatrixR, distR)
    Undistort('imgHorCAMR{}.png'.format(i), 'imgHorCAMR{}.png'.format(i), cameraMatrixR, distR)

    Undistort('imgVertINVCAML{}.png'.format(i), 'imgVertINVCAML{}.png'.format(i), cameraMatrixL, distL)
    Undistort('imgHorINVCAML{}.png'.format(i), 'imgHorINVCAML{}.png'.format(i), cameraMatrixL, distL)
    Undistort('imgVertINVCAMR{}.png'.format(i), 'imgVertINVCAMR{}.png'.format(i), cameraMatrixR, distR)
    Undistort('imgHorINVCAMR{}.png'.format(i), 'imgHorINVCAMR{}.png'.format(i), cameraMatrixR, distR)


## Undistorting 4 Threshold images
for i in range(2):
    Undistort('ThresholdCAML{}.png'.format(i),'ThresholdCAML{}.png'.format(i),cameraMatrixL,distL)
    Undistort('ThresholdCAMR{}.png'.format(i),'ThresholdCAMR{}.png'.format(i),cameraMatrixR,distR)