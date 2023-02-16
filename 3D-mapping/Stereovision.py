import os
import cv2 as cv
import pickle
import InputParameters


os.chdir(InputParameters.StereovisionDirectory)
imgL = cv.imread('imgHorCAML0.png',cv.IMREAD_UNCHANGED)
imgR = cv.imread('imgHorCAMR0.png',cv.IMREAD_UNCHANGED)
grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

os.chdir(InputParameters.WorkingDirectory)
with open('CameraMatrixL.pkl', 'rb') as file3:
    newCameraMatrixL = pickle.load(file3)
with open('CameraMatrixR.pkl', 'rb') as file4:
    newCameraMatrixR = pickle.load(file4)
with open('DistortionL.pkl', 'rb') as file5:
    distL = pickle.load(file5)
with open('DistortionR.pkl', 'rb') as file6:
    distR = pickle.load(file6)
with open('RotationMatrix.pkl', 'rb') as file3:
    rot = pickle.load(file3)
    print('Rotation Matrix : ',rot)
with open('TranslationMatrix.pkl', 'rb') as file4:
    trans = pickle.load(file4)



def Rectify():

    rectifyScale= 1
    rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

    stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
    stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

    print("Saving parameters!")
    os.chdir(InputParameters.StereovisionDirectory)
    cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])
    cv_file.write('q',Q)
    cv_file.release()
    return Q



