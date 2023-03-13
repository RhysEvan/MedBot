from Image_processor import *
from Graycode import *
from Triangulation import *
import cv2
from InputParameters import *
import time
from Detection import *

## Gray code Pattern Generation (gets saved)
#pattern = Graycode.GrayCode()
#pattern.BinaireCode()
#pattern.GrayCode()
#pattern.RepmatVertical()
#pattern.RepmatHorizontal()
class Mapping():
    def __init__(self):
        self.handler = Image()

    def intrinsic_calibration(self):
        ## Calibration
        calibrate = input('Do you want to do a new calibration? Yes/no \nIf not, parameters of last calibration will be used. These may be incorrect. \nYes / no : ')
        if calibrate == 'yes' or calibrate == 'Yes':
            print('Starting Calibration...')
            import StereoCalibration
        elif calibrate == 'no' or calibrate == 'No':
            print('Calibration parameters of last calibration used :')
            PrintParameters()

    def original_code(self):
        ## Cameracapture with Pattern Projection from projector
        self.handler.Frame(length)              # Input : number of images taken (dependant on resolution so "Graycode.length" is used)
        self.handler.GetThreshold()
        DecodeGrayCode(binaryMaxValue,methodOfThreshold)
        Gray2Dec()

        ## Triangulation
        PrintParameters()
        Triangulate()

    def continuous_mapping(self):
        self.handler.ContinFrame()
        self.handler.ContinThreshold()
        DecodeGrayCode(binaryMaxValue,methodOfThreshold)
        Gray2Dec()

        ## Triangulation
        PrintParameters()
        Triangulate()


if __name__ == "__main__":
    map = Mapping()
    #map.intrinsic_calibration()
    test_run = True
    if test_run:
        map.original_code