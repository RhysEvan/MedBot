import cv2
import time
import os
import numpy as np
import mapping.Projector as Projector
import mapping.InputParameters as InputParameters
from CameraModel.Pleora.RGB.GenericRGBCamera import GenericRGBCamera
from skimage import img_as_ubyte
import matplotlib.pyplot as plt


class Image_Handle():
    def __init__(self,test = False, location_L = None, location_R = None):
        if test:
            self.cam_L = GenericRGBCamera()
            self.cam_R = GenericRGBCamera()
            self.loc_L = location_L
            self.loc_R = location_R
        elif not test:
            self.emit_camL = []
            self.emit_camR = []

    def OpenCAM (self):
        ret = self.cam_L.Open(self.loc_L)
        if ret == None:
            return
        self.cam_L.Start()
        self.cam_L.SetParameterDouble("ExposureTime", 17000)
        self.cam_L.SetParameterDouble("Gain", 15.5)
        self.cam_L.SetParameterDouble("Binning",2)
        ret = self.cam_R.Open(self.loc_R)
        if ret == None:
            return
        self.cam_R.Start()
        self.cam_R.SetParameterDouble("ExposureTime", 17000)
        self.cam_R.SetParameterDouble("Gain", 15.5)
        self.cam_R.SetParameterDouble("Binning", 2)

    def GetFrame(self):
        self.Image_L = cv2.cvtColor(self.cam_L.GetFrame().clip(0,255).astype(np.uint8), cv2.COLOR_BayerRGGB2GRAY)
        self.Image_R = cv2.cvtColor(self.cam_R.GetFrame().clip(0,255).astype(np.uint8), cv2.COLOR_BayerRGGB2GRAY)
        return [self.Image_L , self.Image_R]

    def single_run_Threshold(self):
        image_counter = 0
        Threshold_list= []
        while True:
            if image_counter < 2:
                Projector.imgToScrn(0, 'Thresholdimage{}.png'.format(image_counter))
                time.sleep(0.17)
                Threshold_list.append(self.GetFrame())
                image_counter += 1
            else:
                break
        Projector.DestroyWindow()
        return np.array(Threshold_list)

    def single_run_frame(self,numberOfImages):
        image_counter = 0
        Vert_list = []
        INV_Vert_list = []
        Horz_list = []
        INV_Horz_list = []
        while True :            
            ## First loop for Vertical patterns
            if image_counter < numberOfImages:
                Projector.imgToScrn(0, 'graycodeVert{}.png'.format(image_counter))
                if image_counter == 0:
                    time.sleep(0.17)                                       # Longer waiting time with first picture for better light adjustment
                else:
                    time.sleep(0.17)
                Vert_list.append(self.GetFrame())

                ## Patterns for invers Gray code
                Projector.imgToScrn(0, 'graycodeVertInv{}.png'.format(image_counter))
                time.sleep(0.17)
                INV_Vert_list.append(self.GetFrame())
                image_counter += 1
            else :
                break
        image_counter = 0
        while True :            ## Second loop for Horizontal patterns
            if image_counter < numberOfImages:
                Projector.imgToScrn(0, 'graycodeHor{}.png'.format(image_counter))
                time.sleep(0.16)
                Horz_list.append(self.GetFrame())

                ## Patterns for invers Gray code
                Projector.imgToScrn(0, 'graycodeHorInv{}.png'.format(image_counter))
                time.sleep(0.16)
                INV_Horz_list.append(self.GetFrame())
                image_counter += 1
            else :
                break
        return [np.array(Vert_list), np.array(INV_Vert_list), np.array(Horz_list), np.array(INV_Horz_list)]

    def ContinThreshold(self):
        image_counter = 0
        while True:
            if image_counter < 2:
                Projector.imgToScrn(0, 'Thresholdimage{}.png'.format(image_counter))
                time.sleep(0.19)
                os.chdir(InputParameters.ImageDirectory)  # Temporary way of saving images in certain file (still needs to be changed)
                cv2.imwrite("ThresholdCAML{}.png".format(image_counter), self.emit_camL)
                cv2.imwrite("ThresholdCAMR{}.png".format(image_counter), self.emit_camR)
                os.chdir(InputParameters.WorkingDirectory)  # Temporary way of saving images in certain file (still needs to be changed)
                image_counter += 1
            else:
                break

    def ContinFrame(self,numberOfImages):
        image_counter = 0
        os.chdir(InputParameters.ImageDirectory)
        while True :            ## First loop for Vertical patterns
            if image_counter < numberOfImages:
                Projector.imgToScrn(0, 'graycodeVert{}.png'.format(image_counter))
                if image_counter == 0:
                    time.sleep(0.3)                                       # Longer waiting time with first picture for better light adjustment
                else:
                    time.sleep(0.3)
                cv2.imwrite("imgVertCAML{}.png".format(image_counter), self.emit_camL)
                cv2.imwrite("imgVertCAMR{}.png".format(image_counter), self.emit_camR)

                ## Patterns for invers Gray code
                Projector.imgToScrn(0, 'graycodeVertInv{}.png'.format(image_counter))
                time.sleep(0.3)
                cv2.imwrite("imgVertINVCAML{}.png".format(image_counter), self.emit_camL)
                cv2.imwrite("imgVertINVCAMR{}.png".format(image_counter), self.emit_camR)
                image_counter += 1
            else :
                break

        image_counter = 0
        while True :            ## Second loop for Horizontal patterns
            if image_counter < numberOfImages:
                Projector.imgToScrn(0, 'graycodeHor{}.png'.format(image_counter))
                time.sleep(0.3)
                cv2.imwrite("imgHorCAML{}.png".format(image_counter) , self.emit_camL)
                cv2.imwrite("imgHorCAMR{}.png".format(image_counter), self.emit_camR)

                ## Patterns for invers Gray code
                Projector.imgToScrn(0, 'graycodeHorInv{}.png'.format(image_counter))
                time.sleep(0.3)
                cv2.imwrite("imgHorINVCAML{}.png".format(image_counter), self.emit_camL)
                cv2.imwrite("imgHorINVCAMR{}.png".format(image_counter), self.emit_camR)

                image_counter += 1
            else :
                break




