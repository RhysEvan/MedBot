from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2


class Feed_Left(QThread):
    def __init__(self , location = None, parent=None):
        super(Feed_Left,self).__init__(parent)
        self.loc = location
    ImageUpdateLeft = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(self.loc)
        while self.ThreadActive:
            ret, frame = cap.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdateLeft.emit(Pic)
    def stop(self):
        self.ThreadActive = False
        self.quit()

class Feed_Right(QThread):
    def __init__(self , location = None, parent=None):
        super(Feed_Right,self).__init__(parent)
        self.loc = location
    ImageUpdateRight = pyqtSignal(QImage)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(self.loc)
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdateRight.emit(Pic)
    def stop(self):
        self.ThreadActive = False
        self.quit()