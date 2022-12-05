from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import numpy as np
import cv2
from CameraModel.Pleora.RGB.GenericRGBCamera import GenericRGBCamera

class Feed(QThread):
    def __init__(self , location = None, parent=None):
        super(Feed,self).__init__(parent)
        self.cam = GenericRGBCamera()
        self.loc = location
    ImageUpdate= pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True 
        try: 
            self.cam.Open(self.loc)
            self.cam.Start()
        except ValueError:
            print("no camera")
            return
        self.cam.SetParameterDouble("ExposureTime", 3000)
        self.cam.SetParameterDouble("Gain", 24)

        while self.ThreadActive:
            self.Image = self.cam.GetFrame()
            self.Image = cv2.cvtColor(self.Image.astype(np.uint16), cv2.COLOR_BayerRGGB2RGB)
            self.Image[...,1] = (self.Image[...,1]*0.5)#.astype(np.uint8)
            self.Image[...,0] = (self.Image[...,0]*1)#.astype(np.uint8)
            self.Image[...,2] = (self.Image[...,2]*1)#.astype(np.uint8)
            self.Image = cv2.resize(self.Image, [640,480])
            self.Image = self.Image.astype(np.uint8)
            if self.Image is not None:
                ConvertToQtFormat = QImage(self.Image.data, self.Image.shape[1], self.Image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
            else:
                self.cam.Close()
                time.sleep(1)
                ret = self.cam.Open(self.loc)

    def stop(self):
        self.ThreadActive = False
        self.cam.Close()
        self.quit()