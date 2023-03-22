from mapping.Image_processor import Image_Handle
from mapping.Graycode import *
from mapping.Triangulation import triangulator
import cv2
from mapping.InputParameters import *
import time
from mapping.Detection import Detecting


## Gray code Pattern Generation (gets saved)
#pattern = Graycode.GrayCode()
#pattern.BinaireCode()
#pattern.GrayCode()
#pattern.RepmatVertical()
#pattern.RepmatHorizontal()

class Mapping():
    def __init__(self):
        self.cont_handler = Image_Handle()
        self.detect = Detecting(self)
        self.triang = triangulator(self)
        self.test = False

        if __name__ == "__main__":
                self.handler = Image_Handle(test = True, location_L= InputParameters.LeftCamera, location_R= InputParameters.RightCamera)
                self.handler.OpenCAM()

    def intrinsic_calibration(self):
        ## Calibration
        calibrate = input('Do you want to do a new calibration? Yes/no \nIf not, parameters of last calibration will be used. These may be incorrect. \nYes / no : ')
        if calibrate == 'yes' or calibrate == 'Yes':
            print('Starting Calibration...')
            import mapping.StereoCalibration
        elif calibrate == 'no' or calibrate == 'No':
            print('Calibration parameters of last calibration used :')
            self.triang.PrintParameters()
    
    def map_test_code(self):
        start = time.time()
        self.Vert_list, self.INV_Vert_list, self.Horz_list, self.INV_Horz_list = self.handler.single_run_frame(length)
        self.Threshold_list = self.handler.single_run_Threshold()
        end = time.time()
        print(end-start)
        self.detect.DecodeGrayCode(binaryMaxValue)
        self.detect.Gray2Dec()

        ## Triangulation
        self.triang.PrintParameters()
        self.triang.Triangulate()
        
    def continuous_mapping(self):
        self.cont_handler.ContinFrame()
        self.cont_handler.ContinThreshold()
        self.detect.DecodeGrayCode(binaryMaxValue)
        self.detect.Gray2Dec()

        ## Triangulation
        self.triang.PrintParameters()
        self.triang.Triangulate()

if __name__ == "__main__":
    map = Mapping()
    #map.intrinsic_calibration()
    test_run = True
    if test_run:
        map.test = True
        map.map_test_code()
        