from Image_processor import Image_Handle
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
        self.cont_handler = Image_Handle()

        if __name__ == "__main__":
                self.handler = Image_Handle(test = True, location_L= InputParameters.LeftCamera, location_R= InputParameters.RightCamera)
                self.handler.OpenCAM()

    def intrinsic_calibration(self):
        ## Calibration
        calibrate = input('Do you want to do a new calibration? Yes/no \nIf not, parameters of last calibration will be used. These may be incorrect. \nYes / no : ')
        if calibrate == 'yes' or calibrate == 'Yes':
            print('Starting Calibration...')
            import StereoCalibration
        elif calibrate == 'no' or calibrate == 'No':
            print('Calibration parameters of last calibration used :')
            PrintParameters()
    
    def map_test_code(self):
        start = time.time()
        Vert_list, INV_Vert_list, Horz_list, INV_Horz_list = self.handler.single_run_frame(length)
        Threshold_list = self.handler.single_run_Threshold()
        end = time.time()
        print(end-start)
        DecodeGrayCode(binaryMaxValue,Threshold_list,Vert_list, INV_Vert_list, Horz_list, INV_Horz_list)
        Gray2Dec()

        ## Triangulation
        PrintParameters()
        Triangulate()
        

    def continuous_mapping(self):
        self.cont_handler.ContinFrame()
        self.cont_handler.ContinThreshold()
        DecodeGrayCode(binaryMaxValue)
        Gray2Dec()

        ## Triangulation
        PrintParameters()
        Triangulate()


if __name__ == "__main__":
    map = Mapping()
    #map.intrinsic_calibration()
    test_run = True
    if test_run:
        map.map_test_code()