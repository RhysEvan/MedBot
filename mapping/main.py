import Camera1
import Graycode
import Triangulation
import cv2
import InputParameters
import time
import Detection

## Gray code Pattern Generation (gets saved)
#pattern = Graycode.GrayCode()
#pattern.BinaireCode()
#pattern.GrayCode()
#pattern.RepmatVertical()
#pattern.RepmatHorizontal()
class Mapping():
    def __init__(self):
        self.cam = Camera1.Webcam()

    def intrinsic_calibration(self):
        ## Calibration
        calibrate = input('Do you want to do a new calibration? Yes/no \nIf not, parameters of last calibration will be used. These may be incorrect. \nYes / no : ')
        try:
            if calibrate == 'yes' or calibrate == 'Yes':
                print('Starting Calibration...')
                import StereoCalibration

            elif calibrate == 'no' or calibrate == 'No':
                print('Calibration parameters of last calibration used :')
                Triangulation.PrintParameters()

        except:
            print('Please type yes/no for calibration')

    def original_code(self):
        ## Cameracapture with Pattern Projection from projector
        self.cam.GetFrame(Graycode.length)              # Input : number of images taken (dependant on resolution so "Graycode.length" is used)
        self.cam.GetThreshold()
        Detection.DecodeGrayCode(InputParameters.binaryMaxValue,InputParameters.methodOfThreshold)
        Detection.Gray2Dec()

        ## Triangulation
        Triangulation.PrintParameters()
        Triangulation.Triangulate()


    def continuous_mapping(self):
        self.cam.ContinThreshold()
        Detection.DecodeGrayCode(InputParameters.binaryMaxValue,InputParameters.methodOfThreshold)
        Detection.Gray2Dec()

        ## Triangulation
        Triangulation.PrintParameters()
        Triangulation.Triangulate()


if __name__ == "__main__":
    map = Mapping()
    map.intrinsic_calibration
    test_run = False
    if test_run:
        map.original_code