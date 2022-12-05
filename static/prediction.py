from Retna.main import Main
from static.Colour_detection import Colour_detect

import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class prediction():
    def __init__(self, main):
        self.retna = Main
        self.gui = main
        self.col_detect = Colour_detect
    
    def show_predict(self):
        
        R_im = self.gui.cam_r.Image
        contours = self.col_detect.wound_encase(R_im)
        R_crop = self.cropper(R_im, contours)
        R_pred = self.retna.cam_predict(R_crop, (self.main.path+r"\Retna\models\checkpoint.pt"))
        ConvertToQtFormat = QImage(R_pred.data, R_pred.shape[1], R_pred.shape[0], QImage.Format_RGB888)
        R_Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.gui.predict_right.setPixmap(QPixmap.fromImage(R_Pic))
        
        L_im = self.gui.cam_l.Image
        contours = self.col_detect.wound_encase(L_im)
        L_crop = self.cropper(L_im, contours)
        L_pred = self.retna.cam_predict(L_crop, (self.main.path+r"\Retna\models\checkpoint.pt"))
        ConvertToQtFormat = QImage(L_pred.data, L_pred.shape[1], L_pred.shape[0], QImage.Format_RGB888)
        L_Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.gui.predict_right.setPixmap(QPixmap.fromImage(L_Pic))
    
    def cropper(self,im,contours):
        for im, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 200):
                x_skin, y_skin, w_skin, h_skin = cv2.boundingRect(contour)
                in_crop = im[:,x_skin:(x_skin+w_skin), y_skin:(y_skin+h_skin)]
        return in_crop
