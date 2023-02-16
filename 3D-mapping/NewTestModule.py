import os
import shutil
import cv2 as cv
import InputParameters

for i in range(1):
    os.chdir(InputParameters.CalibratedImageDirectory)
    img = cv.imread('imgVertCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
    os.chdir(InputParameters.imagesLeft)
    cv.imwrite('imgVertCAML{}.png'.format(i),img)