from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import numpy as np
import cv2
from skimage import img_as_ubyte
from CameraModel.Pleora.RGB.GenericRGBCamera import GenericRGBCamera
from PIL import Image
from PIL.ImageQt import ImageQt
from static.prediction import prediction


class Feed(QThread):
    def __init__(self , location = None, parent=None):
        super(Feed,self).__init__(parent)
        self.cam = GenericRGBCamera()
        self.loc = location
        self.predict = prediction()
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
        self.cam.SetParameterDouble("ExposureTime",10000)
        self.cam.SetParameterDouble("Gain", 15.5)
        self.cam.SetParameterDouble("Binning", 2)
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
        self.cam.SetParameterDouble("ExposureTime", 10000)
        self.cam.SetParameterDouble("Gain", 15.5)
        self.cam.SetParameterInteger("BinningHorizontal",2)
        self.cam.SetParameterInteger("BinningVertical",2)
        self.calib_percentile_whitebalance(99)
        then = 0
        while self.ColorActive:
            now = time.time()
            if (now-then) > 0.1:
                if self.ret is not None:
                    Pic = self.RGBFrame()
                    if Pic is not None:
                        self.ImageUpdate.emit(Pic)
                        then = time.time()
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
            Pic = self.white_balance_image(self.Image,self.Calib)
            img = self.predict.paste_predict(Pic)
            img = Image.fromarray(img)
            Pic = ImageQt(img)
            self.first = False
            return Pic
        else:
            return None

    def white_balance_image(self, image, Calib):
        return img_as_ubyte((image * 1.0 / Calib).clip(0, 1))

    def calib_percentile_whitebalance(self, percentile_value):
        image = self.cam.GetFrame()
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BayerRGGB2RGB)
        self.Calib =  np.percentile(image, percentile_value, axis=(0, 1))