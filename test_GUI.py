from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import sys
import numpy as np
from typing import Dict

from colour_GUI import Ui_MainWindow
from serial_com import serial_bridge
from CameraModel.Pleora.RGB.GenericRGBCamera import GenericRGBCamera
from CameraModel.Pleora.PleoraCamera import *

class Feed(QThread):
    def __init__(self , location = None, parent=None):
        super(Feed,self).__init__(parent)
        self.cam = GenericRGBCamera()
        self.loc = location

    ImageUpdate= pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        self.cam.Open("2BA200004267")
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
                self.cam.Open(self.loc)

    def stop(self):
        self.ThreadActive = False
        self.quit()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("My App")
        self.com = serial_bridge()
        self.cam = Feed(0)
        self.cam.start()
        self.disx = '0'
        self.disy = '0'
        self.disd = '0'
        self.cam.ImageUpdate.connect(self.illustrate)
        self.distance.clicked.connect(self.distance_calc)
        self.sending.clicked.connect(self.sender)
    def illustrate(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))    
    def distance_calc(self):
        [self.disx, self.disy, self.disd]= self.cam.colouring.calc()
        print(self.disx)
        print(self.disy)
        print(self.disd)
        print("add implementation that states that if colour becomes invisble it returns the print 'open gripper'.")
    def sender(self):
        print("sending")
        self.com.home()
        # I will be assuming some math parameters and I will try and make a pre-emtive version of the camera calculations.
        self.com.send_move(self.disx)
        self.com.send_move(self.disy)
        self.com.send_move(self.disd)




app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()