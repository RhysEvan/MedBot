from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import numpy as np

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
        self.cam.SetParameterDouble("Gain", 10)

        while self.ThreadActive:
            Image = self.cam.GetFrame()
            if Image is not None:
                Image = Image.astype(np.uint8)
                ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
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