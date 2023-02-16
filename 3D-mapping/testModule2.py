import cv2
import numpy as np
import os
import InputParameters

def nothing(x):
         pass

     # Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

    # create trackbars for color change
cv2.createTrackbar('value1','image',4,150,nothing)
cv2.createTrackbar('value2','image',0,150,nothing)
cv2.createTrackbar('Threshold','image',0,255,nothing)




while(1):
   cv2.imshow('image',img)
   k = cv2.waitKey(1) & 0xFF
   if k == 27:
      break

        # get current positions of four trackbars
   r1 = cv2.getTrackbarPos('value1','image')
   r = int(r1 * 2 + 1)
   g1 = cv2.getTrackbarPos('value2','image')
   g = int(g1 * 2 + 1)
   thresh = cv2.getTrackbarPos('Threshold','image')



   os.chdir(InputParameters.CalibratedImageDirectory)
   imageAllBlack = cv2.imread('imgVertCAML8.png', cv2.IMREAD_UNCHANGED)
   greyscaleAllBlack = cv2.cvtColor(imageAllBlack, cv2.COLOR_BGR2GRAY)
   imgBlackL = cv2.adaptiveThreshold(greyscaleAllBlack, thresh, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, r,g1)
   imS = cv2.resize(imgBlackL, (1080, 1400))
   cv2.imshow('adaptive',imgBlackL)

cv2.destroyAllWindows()