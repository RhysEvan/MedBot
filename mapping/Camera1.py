import cv2
import time
import os
from .Projector import *
from .InputParameters import *

class Webcam:
    def __init__(self):
        self.emit_camL = []
        self.emit_camR = []

    def OpenCAML (self,nr):
        vid = cv2.VideoCapture(nr)      # (nr,cv2.CAP_DSHOW) to get rid off error, but then camera caputers to soon
        self.vid = vid

    def OpenCAMR(self,nr):
        vid2 = cv2.VideoCapture(nr)     # (nr,cv2.CAP_DSHOW) to get rid off error, but then camera caputers to soon
        self.vid2 = vid2

    def GetFrame (self,numberOfImages):
        Webcam.OpenCAML(self,LeftCamera)
        Webcam.OpenCAMR(self,RightCamera)
        cam1 = self.vid
        cam2 = self.vid2
        image_counter = 0

        os.chdir(ImageDirectory)
        while True :            ## First loop for Vertical patterns
            if image_counter < numberOfImages:
                imgToScrn(0, 'graycodeVert{}.png'.format(image_counter))
                if image_counter == 0:
                    time.sleep(0.3)                                       # Longer waiting time with first picture for better light adjustment
                else:
                    time.sleep(0.3)
                ret, frame1 = cam1.read()
                ret, frame2 = cam2.read()
                cv2.imwrite("imgVertCAML{}.png".format(image_counter), frame1)
                cv2.imwrite("imgVertCAMR{}.png".format(image_counter), frame2)

                ## Patterns for invers Gray code
                imgToScrn(0, 'graycodeVertInv{}.png'.format(image_counter))
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
                imgToScrn(0, 'graycodeHor{}.png'.format(image_counter))
                time.sleep(0.3)
                ret, frame = cam1.read()
                ret, frame2 = cam2.read()
                cv2.imwrite("imgHorCAML{}.png".format(image_counter) , frame)
                cv2.imwrite("imgHorCAMR{}.png".format(image_counter), frame2)

                ## Patterns for invers Gray code
                imgToScrn(0, 'graycodeHorInv{}.png'.format(image_counter))
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
        Webcam.OpenCAML(self, InputParameters.LeftCamera)
        Webcam.OpenCAMR(self, InputParameters.RightCamera)
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
                imgToScrn(0, 'Thresholdimage{}.png'.format(image_counter))
                time.sleep(0.19)
                frame = self.emit_camL
                frame2 = self.emit_camR
                os.chdir(ImageDirectory)  # Temporary way of saving images in certain file (still needs to be changed)
                cv2.imwrite("ThresholdCAML{}.png".format(image_counter), frame)
                cv2.imwrite("ThresholdCAMR{}.png".format(image_counter), frame2)
                os.chdir(WorkingDirectory)  # Temporary way of saving images in certain file (still needs to be changed)
                image_counter += 1
            else:
                break
        cv2.destroyAllWindows()
        DestroyWindow()




