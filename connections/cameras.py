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
    ImageUpdate= pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True 
        try: 
            self.cam.Open(self.loc)
            self.cam.Start()
        except ValueError:
            print("no camera")
            return
        self.cam.SetParameterDouble("ExposureTime", 2000)
        self.cam.SetParameterDouble("Gain", 12)
        self.calib_percentile_whitebalance(99)
        while self.ThreadActive:
            self.Image = self.cam.GetFrame()
            self.Image = cv2.cvtColor(self.Image.astype(np.uint16), cv2.COLOR_BayerRGGB2RGB)
            self.Image = self.white_balance_image(self.Image,self.Calib)
            self.Image = self.Image.astype(np.uint8)
            if self.Image is not None:
                ConvertToQtFormat = QImage(self.Image.data, self.Image.shape[1], self.Image.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(1536, 2048, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
            else:
                self.cam.Close()
                time.sleep(1)
                ret = self.cam.Open(self.loc)

    def stop(self):
        self.ThreadActive = False
        self.cam.Close()
        self.quit()


    def white_balance_image(self, image, Calib):
        return img_as_ubyte((image * 1.0 / Calib).clip(0, 1))

    def calib_percentile_whitebalance(self, percentile_value):
        image = self.cam.GetFrame()
        image = cv2.cvtColor(image.astype(np.uint16), cv2.COLOR_BayerRGGB2RGB)
        self.Calib =  np.percentile(image, percentile_value, axis=(0, 1))