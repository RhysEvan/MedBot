import Retna_run_training
import numpy as np
from skimage.morphology import skeletonize, thin
from skimage.transform import resize
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib.pyplot as plt

class prediction():
    def __init__(self):
        self.retna = Retna_run_training.Main("","",[])
        self.retna.load_state_dict(r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis\Retna\models", "\\pleora_state_dict_2.pt")

    def paste_predict(self, image):
        if image is None:
            return
        else:
            #contours = self.wound_encase(image)
            #crop = self.cropper(image, contours)
            pred = self.retna.cam_predict(image)
            #skeleton = pred
            skeleton = skeletonize(pred)
            image = (resize(image, (480,640))*255).astype(np.uint8)
            #image = np.tile(image[...,None],[1,1,3]).astype(np.uint8)
            skeleton =  (resize(skeleton, image.shape)*255).astype(np.uint8)
            image[skeleton>100] = skeleton[skeleton >100]
            return image

    def cropper(self,im,contours):
        for im, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            print(area)
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