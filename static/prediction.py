from Retna.main import Main
from PIL import Image
import numpy as np
from skimage.morphology import skeletonize
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class prediction():
    def __init__(self, main):
        self.retna = Main
        self.gui = main
    
    def show_predict(self):
        print("currently connected to print predict left call, make sure that trigger is eventually removed since it won't need a button")
        R_im = self.gui.cam_r.Image
        contours = self.wound_encase(R_im)
        R_crop = self.cropper(R_im, contours)
        R_pred = self.retna.cam_predict(R_crop, (r".\Retna\models\checkpoint.pt"))
        ConvertToQtFormat = QImage(R_pred.data, R_pred.shape[1], R_pred.shape[0], QImage.Format_RGB888)
        R_Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.gui.predict_right.setPixmap(QPixmap.fromImage(R_Pic))
        
        L_im = self.gui.cam_l.Image
        contours = self.wound_encase(L_im)
        L_crop = self.cropper(L_im, contours)
        L_pred = self.retna.cam_predict(L_crop, (r".\Retna\models\checkpoint.pt"))
        ConvertToQtFormat = QImage(L_pred.data, L_pred.shape[1], L_pred.shape[0], QImage.Format_RGB888)
        L_Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
        self.gui.predict_right.setPixmap(QPixmap.fromImage(L_Pic))
    
    def paste_predict(self, image):
        if image == None:
            return
        else:
            contours = self.wound_encase(image)
            crop = self.cropper(image, contours)
            pred = self.retna.cam_predict(crop, (r".\Retna\models\checkpoint.pt"))
            skeleton = skeletonize(pred)
            image.paste(skeleton, (self.x_skin, self.y_skin))
            return image

    def cropper(self,im,contours):
        for im, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 200):
                self.x_skin, self.y_skin, w_skin, h_skin = cv2.boundingRect(contour)
                in_crop = im[:,self.x_skin:(self.x_skin+w_skin), self.y_skin:(self.y_skin+h_skin)]
        return in_crop

    def wound_encase(self, imageFrame):
        print("colour values still need to be changed!")
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
        red_lower = np.array([150, 120, 160], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
        kernal = np.ones((5, 5), "uint8")
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(imageFrame, imageFrame, 
                                mask = red_mask)
        contours, hierarchy = cv2.findContours(red_mask,
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
        return contours         