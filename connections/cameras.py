from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import numpy as np
import cv2
from skimage import img_as_ubyte
from CameraModel.Pleora.RGB.GenericRGBCamera import GenericRGBCamera

class Feed(QThread):
    def __init__(self , location = None, parent=None):
        super(Feed,self).__init__(parent)
        self.cam = GenericRGBCamera()
        self.loc = location
        self.first = True
        self.ColorActive = True
        self.MappingActive = False
    ImageUpdate= pyqtSignal(QImage)

    def run(self):
        self.ret = self.cam.Open(self.loc)
        if self.ret == None:
            return
        self.Color_run()

    def Mapping_run(self):
        self.cam.SetParameterDouble("ExposureTime",800)
        self.cam.SetParameterDouble("Gain", 10)
        while self.MappingActive:
            if self.ret is not None:
                Grey = self.GREYFrame()
                if Grey:
                    self.ImageUpdate.emit(Grey)
                else:
                    self.cam.Close()
                    self.reconnect()

    def Color_run(self): 
        self.cam.Start()
        self.cam.SetParameterDouble("ExposureTime", 1000)
        self.cam.SetParameterDouble("Gain", 10)
        self.calib_percentile_whitebalance(99)
        while self.ColorActive:
            if self.ret is not None:
                Pic = self.RGBFrame()
                if Pic:
                    self.ImageUpdate.emit(Pic)
                else:
                    self.cam.Close()
                    self.reconnect()
                    
    def reconnect(self):
        while True:
            time.sleep(1)
            ret = self.cam.Open(self.loc)
            if ret is not None:
                self.first = True
                break

    def stop(self):
        self.ThreadActive = False
        self.cam.Close()
        self.quit()
    
    def GREYFrame(self):
        Pic = cv2.cvtColor(self.cam.GetFrame().clip(0,255).astype(np.uint8), cv2.COLOR_BayerRGGB2GRAY)
        return Pic
    
    def RGBFrame(self):
        self.Image = self.cam.GetFrame()
        if not (self.Image == 0).all() or self.first == True:
            self.Image = cv2.cvtColor(self.Image.astype(np.uint8), cv2.COLOR_BayerRGGB2RGB)
            self.Image = self.white_balance_image(self.Image,self.Calib)
            ConvertToQtFormat = QImage(self.Image.data, self.Image.shape[1], self.Image.shape[0], QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.first = False
            return Pic
        else:
            return None

    def white_balance_image(self, image, Calib):
        return img_as_ubyte((image * 1.0 / Calib).clip(0, 1))

    def calib_percentile_whitebalance(self, percentile_value):
        image = self.cam.GetFrame()
        image = cv2.cvtColor(image.astype(np.uint16), cv2.COLOR_BayerRGGB2RGB)
        self.Calib =  np.percentile(image, percentile_value, axis=(0, 1))