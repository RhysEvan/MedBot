import os
import Graycode
import InputParameters
import cv2 as cv
import numpy as np
import numpy.ma as ma
import time
import matplotlib.pyplot as plt
from operator import and_

class Detecting():
    def __init__(self, maps):
        self.parent = maps
        self.outputThreshold = 0
        self.greyscaleAllWhite = []
        self.greyscaleAllBlack = []
        self.BinaryAllWhite = []
        self.BinaryAllBlack = []

    ## Function to show binary images to set Threshold value. Gets called with cv.CreateTrackbar
    def on_change(self, value):
        ##Read white image Left camera

        imageAllWhite = self.parent.Threshold_list[0][0]
        imageAllWhite = cv.resize(imageAllWhite, (480,640))
        imgWhiteL = cv.threshold(imageAllWhite, value, 255, cv.THRESH_BINARY)
        imgWhiteL = np.asarray(imgWhiteL[1])
        ##Read black image Left camera
        imageAllBlack = self.parent.Threshold_list[1][0]
        imageAllBlack = cv.resize(imageAllBlack, (480,640))
        imgBlackL = cv.threshold(imageAllBlack, value, 255, cv.THRESH_BINARY)
        imgBlackL = np.asarray(imgBlackL[1])

        ##stacking to show all binary images in one window simultaneously
        img = np.hstack((imgWhiteL, imgBlackL))
        cv.imshow('Shadow Mask', img)
        ## Making threshold a function variable to acces later
        self.outputThreshold = int(cv.getTrackbarPos('Threshold', 'Shadow Mask'))

    ## Finding mask array to delete shadow pixels. This mask array is chosen by visualizing binary images where threshold can be adjusted to find optimal threshold value.
    def ShadowMask(self):
        if self.parent.test == True:
            print('Please adjust threshold parameter for shadowmask. Make sure only the areas that are illuminated are seen. When ready press any key to quit.')
            cv.namedWindow('Shadow Mask')
            cv.createTrackbar('Threshold', 'Shadow Mask', 0, 255,self.on_change)
            cv.waitKey(0)
            cv.destroyAllWindows()
            threshold = self.outputThreshold
            print('Shadow mask Threshold = ', threshold)
        for i in range(len(self.parent.Threshold_list)):
            self.greyscaleAllWhite.append(self.parent.Threshold_list[0][i])
            self.greyscaleAllBlack.append(self.parent.Threshold_list[1][i])
            ret, image_binaryAllWhite = cv.threshold(self.greyscaleAllWhite[i], threshold, 255, cv.THRESH_BINARY)
            ret, image_binaryAllBlack = cv.threshold(self.greyscaleAllBlack[i], threshold, 255, cv.THRESH_BINARY)
            self.BinaryAllWhite.append(image_binaryAllWhite)
            self.BinaryAllBlack.append(image_binaryAllBlack)
        ShadowmaskL = np.empty(self.parent.Vert_list[0][0].shape, dtype=object)
        ShadowmaskL = image_binaryAllWhite[0][0] == image_binaryAllBlack[1][0]  ## All pixels where binaryWhite = binaryBlack are not used.
        ShadowmaskR = np.empty(self.parent.Vert_list[0][0].shape, dtype=object)
        ShadowmaskR = image_binaryAllWhite[0][1] == image_binaryAllBlack[1][1]  ## All pixels where binaryWhite = binaryBlack are not used.

        return ShadowmaskL,ShadowmaskR,threshold

    ## Function used in method 5 of DecodeGrayCode for using mean value per pixel thresholding
    def findingThresholdShadowMask(self):
        ThresholdArray = np.add(self.greyscaleAllWhite[0][0], self.greyscaleAllBlack[0][1])
        ThresholdArrayL = ThresholdArray / 2

        ThresholdArray = np.add(self.greyscaleAllWhite[1][0], self.greyscaleAllBlack[1][1])
        ThresholdArrayR = ThresholdArray / 2

        return ThresholdArrayL,ThresholdArrayR

    ## Different algorithms for decoding structured light. Choosing between methods is possible with input methodParameter.
    def DecodeGrayCode(self, binaryMaxValue):
        numberOfImages = InputParameters.numberOfImages      ## number of higher frequency patterns that are not used. Limited resolution can mean less patterns results in higher resolution

        ## Getting shadowmask images
        self.maskX, self.maskY, self.threshold = self.ShadowMask()

        ## Finding threshold through mean. Gets used in method 5 for per-pixel mean value thresholding
        self.ThresholdL, self.ThresholdR = self.findingThresholdShadowMask()

        ################################################# DIFFERENT DECODING METHODS ###############################################################
        start = time.time()

        #OLD METHOD.PY has all the methods that were not used for this thesis
        ArrayVertLWithoutMask = np.full(self.parent.Vert_list[0][0].shape, "", dtype=object)
        ArrayHorLWithoutMask = np.full(self.parent.Horz_list[0][0].shape, "", dtype=object)
        ArrayVertRWithoutMask = np.full(self.parent.Vert_list[0][1].shape, "", dtype=object)
        ArrayHorRWithoutMask = np.full(self.parent.Horz_list[0][1].shape, "", dtype=object)

        for i in range(Graycode.length - numberOfImages):  ## -1 because resolution of projector is higher than camera's
            start2 = time.time()
            for j in range(2):
                ## array for Vertical Patterns
                image_greyscaleVert = self.parent.Vert_list[i][j]
                image_greyscaleVertINV = self.parent.INV_Vert_list[i][j]
                image_greyscaleVert[image_greyscaleVert >= image_greyscaleVertINV] = 1
                image_greyscaleVert[image_greyscaleVert > 1] = 0
                image_binaryVert = image_greyscaleVert.astype('str')
                if j == 0:
                    ArrayVertLWithoutMask = np.add(ArrayVertLWithoutMask, image_binaryVert)
                elif j == 1:
                    ArrayVertRWithoutMask = np.add(ArrayVertRWithoutMask, image_binaryVert)
                ## Array for Horizontal patterns
                image_greyscaleHor = self.parent.Horz_list[i][j]
                image_greyscaleHorINV = self.parent.INV_Horz_list[i][j]
                image_greyscaleHor[image_greyscaleHor >= image_greyscaleHorINV] = 1
                image_greyscaleHor[image_greyscaleHor > 1] = 0
                image_binaryHor = image_greyscaleHor.astype('str')
                if j == 0:
                    ArrayHorLWithoutMask = np.add(ArrayHorLWithoutMask, image_binaryHor)
                elif j == 1:
                    ArrayHorRWithoutMask = np.add(ArrayHorRWithoutMask, image_binaryHor)
                
            end2 = time.time()
            print("time for a picture: "+str(end2-start2))
            print(i,'done')

        ## If no mask is needed. or masking cannot be done correctly because of reflective surface set nomask to True. This way no mask will be added
        nomask = False
        if nomask == True:
            self.maskX = np.zeros_like(ArrayVertLWithoutMask)
            self.maskY= np.zeros_like(ArrayVertRWithoutMask)
        start1 = time.time()
        ################################################# APPLYING MASKS TO DIFFERENT MATRICES FOR LATER USE #########################################################
        arrayVertLMasked = ma.masked_array(ArrayVertLWithoutMask,mask= self.maskX)
        self.arrayVertLMasked = ma.filled(arrayVertLMasked, '0')

        arrayHorLMasked = ma.masked_array(ArrayHorLWithoutMask, mask= self.maskX)
        self.arrayHorLMasked = ma.filled(arrayHorLMasked, '0')

        arrayVertRMasked = ma.masked_array(ArrayVertRWithoutMask, mask= self.maskY)
        self.arrayVertRMasked = ma.filled(arrayVertRMasked, '0')

        arrayHorRMasked = ma.masked_array(ArrayHorRWithoutMask, mask= self.maskY)
        self.arrayHorRMasked = ma.filled(arrayHorRMasked, '0')
        end1 = time.time()
        print("intermideary timer final part: "+str(end1-start1))
        end = time.time()
        totaltime = end - start
        print('Total computing time of method', InputParameters.methodOfTriangulation, ": ", str(totaltime))

    ## Function to get corresponding decimal values for a binary gray code
    def inversegrayCode(self, n):
        inv = 0
        # Taking xor until
        # n becomes zero
        while (n):
            inv = inv ^ n
            n = n >> 1
        return inv

    ## Function to get decimal value of gray code
    def Gray2Dec(self):
        print('Converting Gray to dec code')
        for i in range(self.parent.Vert_list[0][0].shape[0]):
            for j in range(self.parent.Vert_list[0][0].shape[1]):
                self.arrayVertLMasked[i][j] = self.inversegrayCode(int(self.arrayVertLMasked[i][j], 2))
                self.arrayHorLMasked[i][j] = self.inversegrayCode(int(self.arrayHorLMasked[i][j], 2))

        for i in range(self.parent.Horz_list[0][0].shape[0]):
            for j in range(self.parent.Horz_list[0][0].shape[1]):
                self.arrayVertRMasked[i][j] = self.inversegrayCode(int(self.arrayVertRMasked[i][j], 2))
                self.arrayHorRMasked[i][j] = self.inversegrayCode(int(self.arrayHorRMasked[i][j], 2))

    ## Extra function for plotting a gradient picture of the scene.
    def gradientImage(self):
        x = np.array(self.arrayVertLMasked, dtype=float)
        plt.imshow(x, cmap='jet')
        plt.show()

    ##Function to create inverse matrices used for easier correspondence matching. Gets used in Triangulaton()
    def FindCorrespondence(self):

        ## Creating 3D arrays to represent horizontal AND vertical binary codes for each pixel.
        LeftCamera = np.stack((self.arrayVertLMasked, self.arrayHorLMasked), axis=2)
        RightCamera = np.stack((self.arrayVertRMasked, self.arrayHorRMasked), axis=2)

        ## Adjusting view for 3D arrays for easier matrix usage. Both arrays LeftCamera and LeftCameraPleaseWork are identical just differently shown.
        #LeftCameraPleaseWork = LeftCamera.view([(f'f{i}', LeftCamera.dtype) for i in range(LeftCamera.shape[-1])])[
        #   ..., 0].astype('O')
        #RightCameraPleaseWork = RightCamera.view([(f'f{i}', RightCamera.dtype) for i in range(RightCamera.shape[-1])])[
        #   ..., 0].astype('O')
        start = time.time()

        ## Find max values to create empty (0) matrices to use
        MaxValue1 = np.amax(self.arrayVertLMasked)
        MaxValue2 = np.amax(self.arrayHorLMasked)
        MaxValue3 = np.amax(self.arrayVertRMasked)
        MaxValue4 = np.amax(self.arrayHorRMasked)


        InversArrayRowLeft = np.zeros((MaxValue1 + 1,MaxValue2 + 1))
        InversArrayColumnLeft = np.zeros((MaxValue1 + 1,MaxValue2 + 1))
        InversArrayRowRight = np.zeros((MaxValue3 + 1, MaxValue4 + 1))
        InversArrayColumnRight = np.zeros((MaxValue3 + 1, MaxValue4 + 1))

        ##Loop to generate invers matrix of Left Camera matrix for pixel correspondence
        for i in range (self.parent.Vert_list[0][0].shape[0]):
            for j in range (self.parent.Vert_list[0][0].shape[1]):
                indexRowLeft = self.arrayVertLMasked[i][j]
                indexColumnLeft = self.arrayHorLMasked[i][j]
                InversArrayRowLeft[indexRowLeft][indexColumnLeft] = i
                InversArrayColumnLeft[indexRowLeft][indexColumnLeft] = j

        TempMatrixLeft = np.stack((InversArrayColumnLeft, InversArrayRowLeft), axis=2)
        InversMatrixLeftCam = TempMatrixLeft.view([(f'f{i}', TempMatrixLeft.dtype) for i in range(TempMatrixLeft.shape[-1])])[
            ..., 0].astype('O')

        ##Loop to generate invers matrix of Right Camera matrix for pixel correspondence
        for i in range (self.parent.Vert_list[0][1].shape[0]):
            for j in range(self.parent.Vert_list[0][1].shape[1]):
                indexRowRight = self.arrayVertRMasked[i][j]
                indexColumnRight = self.arrayHorRMasked[i][j]
                InversArrayRowRight[indexRowRight][indexColumnRight] = i
                InversArrayColumnRight[indexRowRight][indexColumnRight] = j

        TempMatrixRight = np.stack((InversArrayColumnRight, InversArrayRowRight),axis =2)
        InversMatrixRightCam = TempMatrixRight.view([(f'f{i}', TempMatrixRight.dtype) for i in range(TempMatrixRight.shape[-1])])[
            ..., 0].astype('O')

        end = time.time()
        totaltime = end - start
        print('time transform inverse matrix = ', totaltime)
        return InversMatrixLeftCam,InversMatrixRightCam

