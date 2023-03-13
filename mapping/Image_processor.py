import cv2
import time
import os
import numpy as np
import Projector
import InputParameters
from CameraModel.Pleora.RGB.GenericRGBCamera import GenericRGBCamera
from skimage import img_as_ubyte


class Image():
    def __init__(self, location = None):
        self.cam = GenericRGBCamera()
        self.loc = location
        self.emit_camL = []
        self.emit_camR = []

    def OpenCAM (self):
        ret = self.cam.Open(self.loc)
        if ret == None:
            return
        self.cam.Start()
        self.cam.SetParameterDouble("ExposureTime", 1000)
        self.cam.SetParameterDouble("Gain", 10)
        self.calib_percentile_whitebalance(99)

    def GetFrame(self):
        self.Image = self.cam.GetFrame()
        self.Image = cv2.cvtColor(self.Image.astype(np.uint16), cv2.COLOR_BayerRGGB2RGB)
        self.Image = self.white_balance_image(self.Image,self.Calib)
        self.Raw_Image = self.Image
        self.Image = self.Image.astype(np.uint8)
        return self.Image

    def Frame (self,numberOfImages):
        cam1 = self.vid1
        cam2 = self.vid2
        image_counter = 0

        os.chdir(InputParameters.ImageDirectory)
        while True :            ## First loop for Vertical patterns
            if image_counter < numberOfImages:
                Projector.imgToScrn(0, 'graycodeVert{}.png'.format(image_counter))
                if image_counter == 0:
                    time.sleep(0.3)                                       # Longer waiting time with first picture for better light adjustment
                else:
                    time.sleep(0.3)
                ret, frame1 = cam1.read()
                ret, frame2 = cam2.read()
                cv2.imwrite("imgVertCAML{}.png".format(image_counter), frame1)
                cv2.imwrite("imgVertCAMR{}.png".format(image_counter), frame2)

                ## Patterns for invers Gray code
                Projector.imgToScrn(0, 'graycodeVertInv{}.png'.format(image_counter))
                time.sleep(0.3)
                ret, frame1 = cam1.read()
                ret, frame2 = cam2.read()
                cv2.imwrite("imgVertINVCAML{}.png".format(image_counter), frame1)
                cv2.imwrite("imgVertINVCAMR{}.png".format(image_counter), frame2)

                image_counter += 1
            else :
                break

        image_counter = 0
        while True :            ## Second loop for Horizontal patterns
            if image_counter < numberOfImages:
                Projector.imgToScrn(0, 'graycodeHor{}.png'.format(image_counter))
                time.sleep(0.3)
                ret, frame = cam1.read()
                ret, frame2 = cam2.read()
                cv2.imwrite("imgHorCAML{}.png".format(image_counter) , frame)
                cv2.imwrite("imgHorCAMR{}.png".format(image_counter), frame2)

                ## Patterns for invers Gray code
                Projector.imgToScrn(0, 'graycodeHorInv{}.png'.format(image_counter))
                time.sleep(0.3)
                ret, frame1 = cam1.read()
                ret, frame2 = cam2.read()
                cv2.imwrite("imgHorINVCAML{}.png".format(image_counter), frame1)
                cv2.imwrite("imgHorINVCAMR{}.png".format(image_counter), frame2)

                image_counter += 1
            else :
                break

        #cam1.release()
        #cam2.release()
        #cv2.destroyAllWindows()
        #Projector.DestroyWindow()

    def GetThreshold (self):
        #Webcam.OpenCAML(self, InputParameters.LeftCamera)
        #Webcam.OpenCAMR(self, InputParameters.RightCamera)
        cam1 = self.vid
        cam2 = self.vid2

        image_counter = 0
        while True:
            if image_counter < 2:
                Projector.imgToScrn(0, 'Thresholdimage{}.png'.format(image_counter))
                time.sleep(0.19)
                ret, frame = cam1.read()
                ret, frame2 = cam2.read()
                os.chdir(InputParameters.ImageDirectory)  # Temporary way of saving images in certain file (still needs to be changed)
                cv2.imwrite("ThresholdCAML{}.png".format(image_counter), frame)
                cv2.imwrite("ThresholdCAMR{}.png".format(image_counter), frame2)
                os.chdir(InputParameters.WorkingDirectory)  # Temporary way of saving images in certain file (still needs to be changed)
                image_counter += 1
            else:
                break

        cam1.release()
        cam2.release()
        cv2.destroyAllWindows()
        Projector.DestroyWindow()
    
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


    def white_balance_image(self, image, Calib):
        return img_as_ubyte((image * 1.0 / Calib).clip(0, 1))

    def calib_percentile_whitebalance(self, percentile_value):
        image = self.cam.GetFrame()
        image = cv2.cvtColor(image.astype(np.uint16), cv2.COLOR_BayerRGGB2RGB)
        self.Calib =  np.percentile(image, percentile_value, axis=(0, 1))



