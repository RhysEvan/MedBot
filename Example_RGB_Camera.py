#@defgroup cam_python CAMERA_TOOLBOX_python
#
#These classes are part of the Op3Mech CAMERA toolbox the toolbox is
#available on bitbucket. go to : https://bitbucket.org/SeppeSels/camera_toolbox_python
#
#copyright Seppe Sels Invilab University of Antwerp 03-09-2021
#
#This code is for internal use only (Uantwerpen), please follow CC BY-NC-SA with the additional restriction
#that it cannot be shared outside of the University of Antwerp.
#Bugs, bugfixes and additions to the code need to be reported to Invilab (contact: Seppe Sels)
#for other licences, contact author.
#
#@ingroup cam_python

##White-balancing code: https://jmanansala.medium.com/image-processing-with-python-color-correction-using-white-balancing-6c6c749886dehttps://jmanansala.medium.com/image-processing-with-python-color-correction-using-white-balancing-6c6c749886de

import cv2
import time
from CameraModel.Pleora.RGB.GenericRGBCamera import *
#from DataClass.GenericDataClass import *
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from matplotlib.patches import Rectangle

import os
def plotimage(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


def plotimageCV(img,name='image'):
    cv2.imshow(name, img.astype(np.uint8))
    cv2.waitKey(1)

def white_balance_image(image,Calib):
    return img_as_ubyte((image * 1.0 / Calib).clip(0, 1))

def calib_percentile_whitebalance(image, percentile_value):
    Calib =  np.percentile(image, percentile_value, axis=(0, 1))
    return Calib
try:
    cam = GenericRGBCamera()
    check = cam.Open('GUI')
    #cam.SaveParamsXML('test.xml')
    cam.Start()
    nf = 1000 # number of frames

    ## just show
    for i in range(0, 1):
        frame = cam.GetFrame()
        frame = cv2.cvtColor(frame.astype(np.uint16), cv2.COLOR_BayerRGGB2RGB)
        plotimage(frame)

    Calib = calib_percentile_whitebalance(frame, 99)
    plt.figure()
    new_frame = white_balance_image(frame,Calib)
    plotimage(new_frame)





finally:


    cam.Close()
