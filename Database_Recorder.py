from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import sys
import numpy as np
import os
from PIL import Image

from static.colour_GUI import Ui_MainWindow
try:
    from connections.cameras import *
    pleora_lib = True
except:
    print("Pleora Library not Installed")
    pleora_lib = False

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, pleora = True):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("My App")
        self.path = r"C:\Users\mheva\OneDrive\Bureaublad\temp/"
        self.index = 1
        if pleora:
            ## Threaded Camera Left## 
            self.cam = Feed("2BA200004267") ## Number represents the camera adress on the computer ##

            self.cam.start()
            self.cam.ImageUpdate.connect(self.illustrate)
        self.distance.clicked.connect(self.saver)
        self.sending.clicked.connect(self.recal)
    def illustrate(self, Image):
        self.Image = Image
        self.label.setPixmap(QPixmap.fromImage(Image))
    
    def saver(self):
        self.Image.save(self.path+"Skin_Sample_"+str(self.index)+".tiff","TIFF")
        self.index += 1
    
    def recal(self):
        self.cam.calib_percentile_whitebalance(99)
        print(self.cam.Calib)



app = QApplication(sys.argv)
window = MainWindow(pleora=pleora_lib)
window.show()
app.exec()