from CameraModel.Pleora.RGB.GenericRGBCamera import GenericRGBCamera
from CameraModel.Pleora.PleoraCamera import *

cam = GenericRGBCamera()
cam.Open(0)
for i in range(0,100):
    cam.GetFrame()

cam.Close()