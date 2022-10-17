import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import time

from colour_GUI import Ui_MainWindow

class Colour_detect():
    def __init__(self, imageFrame=None):
        self.image = imageFrame 
           
    def encase(self, imageFrame):
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        # Set range for red color and 
        # define mask
        red_lower = np.array([150, 87, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
        # Set range for green color and 
        # define mask
        green_lower = np.array([50, 100, 100], np.uint8)
        green_upper = np.array([80, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
        # Set range for blue color and
        # define mask
        blue_lower = np.array([94, 80, 2], np.uint8)
        blue_upper = np.array([120, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
        # Morphological Transform, Dilation
        # for each color and bitwise_and operator
        # between imageFrame and mask determines
        # to detect only that particular color
        kernal = np.ones((5, 5), "uint8")
        # For red color
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(imageFrame, imageFrame, 
                                mask = red_mask)
        # For green color
        green_mask = cv2.dilate(green_mask, kernal)
        res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                    mask = green_mask)
        # For blue color
        blue_mask = cv2.dilate(blue_mask, kernal)
        res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                                mask = blue_mask)
        # Creating contour to track red color
        contours, hierarchy = cv2.findContours(red_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                        (x + w, y + h), 
                                        (0, 0, 255), 2)
                  
    
        # Creating contour to track green color
        contours, hierarchy = cv2.findContours(green_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 300):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                        (x + w, y + h),
                                        (0, 255, 0), 2)
                
        return imageFrame


class Feed(QThread):
    def __init__(self , location = None, parent=None):
        super(Feed,self).__init__(parent)
        self.loc = location
        self.colouring = Colour_detect()
    ImageUpdate= pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(self.loc)
        while self.ThreadActive:
            ret, imageFrame = cap.read()
            if ret:
                imageFrame = cv2.flip(imageFrame, 1)
                imageFrame = self.colouring.encase(imageFrame)
                imageFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)
                ConvertToQtFormat = QImage(imageFrame.data, imageFrame.shape[1], imageFrame.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
            else:
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(self.loc)

    def stop(self):
        self.ThreadActive = False
        self.quit()

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("My App")
        self.cam = Feed(0)
        self.cam.start()
        self.cam.ImageUpdate.connect(self.illustrate)

    def illustrate(self, Image):
        self.label.setPixmap(QPixmap.fromImage(Image))

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
