from CameraModel.Pleora.RGB.GenericRGBCamera import GenericRGBCamera
from CameraModel.Pleora.PleoraCamera import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import io
import easygui as eg

class menu():
    def __init__(self):
        self.index = 0
        self.window()
    
    def window(self):
        event = eg.indexbox("scan",["yes","no"])
        if event == 0:
            picture(self.index)
            self.index += 1
        self.window()
def picture(index):
    cam = GenericRGBCamera()
    cam.Open(0)
    cam.Start()
    cam.SetParameterDouble("ExposureTime", 3000)
    cam.SetParameterDouble("Gain", 12)
    os.chdir(r"C:\Users\mheva\OneDrive\Bureaublad\temp")
    im = cam.GetFrame()
    im = cv2.cvtColor(im.astype(np.uint16), cv2.COLOR_BayerRGGB2RGB)
    im[...,0] = (im[...,0]*255/108)#.astype(np.uint8)    
    im[...,1] = (im[...,1]*255/128)#.astype(np.uint8)    
    im[...,2] = (im[...,2]*255/60)#.astype(np.uint8)     
    im = im.clip(0,255)
    im = im.astype(np.uint8)
    io.imsave("skin_data_"+str(index)+".tif", im)
    cam.Close()
def dominant(im):
    pixels = np.float32(im.reshape(-1, 3))

    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    plt.imshow(im)
    plt.show()

    print(dominant)

if __name__ == "__main__":
    menu()