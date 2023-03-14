import os
import Graycode
import InputParameters
import cv2 as cv
import numpy as np
import numpy.ma as ma
import time
import matplotlib.pyplot as plt
from operator import and_

## Getting dimensions of CALIBRATED images
os.chdir(InputParameters.CalibratedImageDirectory)
imgL = cv.imread('imgVertCAML0.png', cv.IMREAD_UNCHANGED)
imgR = cv.imread('imgVertCAMR0.png', cv.IMREAD_UNCHANGED)
widthL = imgL.shape[1]
heightL = imgL.shape[0]
widthR = imgR.shape[1]
heightR = imgR.shape[0]

## Finding mask array to delete shadow pixels. This mask array is chosen by visualizing binary images where threshold can be adjusted to find optimal threshold value.
def ShadowMask():
    os.chdir(InputParameters.CalibratedImageDirectory)

    ## Function to show binary images to set Threshold value. Gets called with cv.CreateTrackbar
    def on_change(value):
        global getThresh
        ##Read white image Left camera
        imageAllWhite = cv.imread('ThresholdCAML0.png', cv.IMREAD_UNCHANGED)
        greyscaleAllWhite = cv.cvtColor(imageAllWhite, cv.COLOR_BGR2GRAY)
        imgWhiteL = cv.threshold(greyscaleAllWhite, value, 255, cv.THRESH_BINARY)
        imgWhiteL = np.asarray(imgWhiteL[1])
        ##Read black image Left camera
        imageAllBlack = cv.imread('ThresholdCAML1.png', cv.IMREAD_UNCHANGED)
        greyscaleAllBlack = cv.cvtColor(imageAllBlack, cv.COLOR_BGR2GRAY)
        imgBlackL = cv.threshold(greyscaleAllBlack, value, 255, cv.THRESH_BINARY)
        imgBlackL = np.asarray(imgBlackL[1])

        ##stacking to show all binary images in one window simultaneously
        img = np.hstack((imgWhiteL, imgBlackL))
        cv.imshow('Shadow Mask', img)
        ## Making threshold a function variable to acces later
        on_change.outputThreshold = int(cv.getTrackbarPos('Threshold', 'Shadow Mask'))

    print('Please adjust threshold parameter for shadowmask. Make sure only the areas that are illuminated are seen. When ready press any key to quit.')
    cv.namedWindow('Shadow Mask')
    cv.createTrackbar('Threshold', 'Shadow Mask', 0, 255,on_change)
    cv.waitKey(0)
    cv.destroyAllWindows()
    threshold = on_change.outputThreshold
    print('Shadow mask Threshold = ', threshold)

    imageAllWhite = cv.imread('ThresholdCAML0.png', cv.IMREAD_UNCHANGED)
    imageAllBlack = cv.imread('ThresholdCAML1.png', cv.IMREAD_UNCHANGED)
    greyscaleAllWhite = cv.cvtColor(imageAllWhite, cv.COLOR_BGR2GRAY)
    greyscaleAllBlack = cv.cvtColor(imageAllBlack, cv.COLOR_BGR2GRAY)
    ret, image_binaryAllWhite = cv.threshold(greyscaleAllWhite, threshold, 255, cv.THRESH_BINARY)
    ret, image_binaryAllBlack = cv.threshold(greyscaleAllBlack, threshold, 255, cv.THRESH_BINARY)
    cv.imwrite("BinaryAllWhiteL.png", image_binaryAllWhite)
    cv.imwrite("BinaryAllBlackL.png", image_binaryAllBlack)
    ShadowmaskL = np.empty((imgL.shape[0], imgL.shape[1]), dtype=object)
    ShadowmaskL = image_binaryAllWhite == image_binaryAllBlack              ## All pixels where binaryWhite = binaryBlack are not used.

    imageAllWhite = cv.imread('ThresholdCAMR0.png', cv.IMREAD_UNCHANGED)
    imageAllBlack = cv.imread('ThresholdCAMR1.png', cv.IMREAD_UNCHANGED)
    greyscaleAllWhite = cv.cvtColor(imageAllWhite, cv.COLOR_BGR2GRAY)
    greyscaleAllBlack = cv.cvtColor(imageAllBlack, cv.COLOR_BGR2GRAY)
    ret, image_binaryAllWhite = cv.threshold(greyscaleAllWhite, threshold, 255, cv.THRESH_BINARY)
    ret, image_binaryAllBlack = cv.threshold(greyscaleAllBlack, threshold, 255, cv.THRESH_BINARY)
    cv.imwrite("BinaryAllWhiteR.png", image_binaryAllWhite)
    cv.imwrite("BinaryAllBlackR.png", image_binaryAllBlack)
    ShadowmaskR = np.empty((imgL.shape[0], imgL.shape[1]), dtype=object)
    ShadowmaskR = image_binaryAllWhite == image_binaryAllBlack              ## All pixels where binaryWhite = binaryBlack are not used.


    return ShadowmaskL,ShadowmaskR,threshold

## Function used in method 5 of DecodeGrayCode for using mean value per pixel thresholding
def findingThresholdShadowMask():
    imageAllWhite = cv.imread('ThresholdCAML0.png', cv.IMREAD_UNCHANGED)
    imageAllBlack = cv.imread('ThresholdCAML1.png', cv.IMREAD_UNCHANGED)
    greyscaleAllWhite = cv.cvtColor(imageAllWhite, cv.COLOR_BGR2GRAY)
    greyscaleAllBlack = cv.cvtColor(imageAllBlack, cv.COLOR_BGR2GRAY)
    ThresholdArray = np.add(greyscaleAllWhite, greyscaleAllBlack)
    ThresholdArrayL = ThresholdArray / 2

    imageAllWhite = cv.imread('ThresholdCAMR0.png', cv.IMREAD_UNCHANGED)
    imageAllBlack = cv.imread('ThresholdCAMR1.png', cv.IMREAD_UNCHANGED)
    greyscaleAllWhite = cv.cvtColor(imageAllWhite, cv.COLOR_BGR2GRAY)
    greyscaleAllBlack = cv.cvtColor(imageAllBlack, cv.COLOR_BGR2GRAY)
    ThresholdArray = np.add(greyscaleAllWhite, greyscaleAllBlack)
    ThresholdArrayR = ThresholdArray / 2

    return ThresholdArrayL,ThresholdArrayR

## Different algorithms for decoding structured light. Choosing between methods is possible with input methodParameter.
def DecodeGrayCode(binaryMaxValue,methodParameter):
    global arrayVertLMasked
    global arrayHorLMasked
    global arrayVertRMasked
    global arrayHorRMasked
    numberOfImages = InputParameters.numberOfImages      ## number of higher frequency patterns that are not used. Limited resolution can mean less patterns results in higher resolution

    ## Getting shadowmask images
    maskX, maskY, threshold = ShadowMask()

    ## Finding threshold through mean. Gets used in method 5 for per-pixel mean value thresholding
    ThresholdL, ThresholdR = findingThresholdShadowMask()

    ################################################# DIFFERENT DECODING METHODS ###############################################################
    print('Processing Taken Images...')
    start = time.time()

    #OLD METHOD.PY has all the methods that were not used for this thesis

    ## 5) Using invers patterns
    if methodParameter == 5:
        ArrayVertLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayHorLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayVertRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        ArrayHorRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        for i in range(Graycode.length - numberOfImages):  ## -1 because resolution of projector is higher than camera's
            os.chdir(InputParameters.CalibratedImageDirectory)

            ## array for Vertical Patterns
            imageVert = cv.imread('imgVertCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            #ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
            imageVertINV = cv.imread('imgVertINVCAML{}.png'.format(i),cv.IMREAD_UNCHANGED)
            image_greyscaleVertINV = cv.cvtColor(imageVertINV, cv.COLOR_BGR2GRAY)
            #ret, image_binaryVertINV = cv.threshold(image_greyscaleVertINV, threshold, binaryMaxValue, cv.THRESH_BINARY)
            image_greyscaleVert[image_greyscaleVert >= image_greyscaleVertINV] = 1
            image_greyscaleVert[image_greyscaleVert > 1] = 0
            image_binaryVert = image_greyscaleVert
            imagebinarytest = image_binaryVert.copy()  ## (Tijdelijk om te zien of zwart wit afbeelding in orde is)
            imagebinarytest[imagebinarytest > 0] = 255

            if i == 6 or 7 or 8 or 5 or 9:
                cv.imwrite("testInversCAML{}.png".format(i), imagebinarytest)
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertLWithoutMask = np.add(ArrayVertLWithoutMask, image_binaryVert)

            ## Array for Horizontal patterns
            imageHor = cv.imread('imgHorCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            #ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
            imageHorINV = cv.imread('imgHorINVCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHorINV = cv.cvtColor(imageHorINV, cv.COLOR_BGR2GRAY)
            #ret, image_binaryHorINV = cv.threshold(image_greyscaleHorINV, threshold, binaryMaxValue, cv.THRESH_BINARY)
            image_greyscaleHor[image_greyscaleHor >= image_greyscaleHorINV] = 1
            image_greyscaleHor[image_greyscaleHor > 1] = 0
            image_binaryHor = image_greyscaleHor

            imagebinarytest = image_binaryHor.copy()  ## (Tijdelijk om te zien of zwart wit afbeelding in orde is)
            imagebinarytest[imagebinarytest > 0] = 255
            if i == 6 or 7 or 8 or 5 or 9:
                cv.imwrite("testInversHORICAML{}.png".format(i), imagebinarytest)

            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorLWithoutMask = np.add(ArrayHorLWithoutMask, image_binaryHor)




            ## Array for Vertical patterns
            imageVert = cv.imread('imgVertCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            #ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
            imageVertINV = cv.imread('imgVertINVCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVertINV = cv.cvtColor(imageVertINV, cv.COLOR_BGR2GRAY)
            #ret, image_binaryVertINV = cv.threshold(image_greyscaleVertINV, threshold, binaryMaxValue, cv.THRESH_BINARY)
            image_greyscaleVert[image_greyscaleVert >= image_greyscaleVertINV] = 1
            image_greyscaleVert[image_greyscaleVert > 1] = 0
            image_binaryVert = image_greyscaleVert
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertRWithoutMask = np.add(ArrayVertRWithoutMask, image_binaryVert)

            ## Array for horizontal patterns
            imageHor = cv.imread('imgHorCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            #ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
            imageHorINV = cv.imread('imgHorINVCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHorINV = cv.cvtColor(imageHorINV, cv.COLOR_BGR2GRAY)
            #ret, image_binaryHorINV = cv.threshold(image_greyscaleHorINV, threshold, binaryMaxValue, cv.THRESH_BINARY)
            image_greyscaleHor[image_greyscaleHor >= image_greyscaleHorINV] = 1
            image_greyscaleHor[image_greyscaleHor > 1] = 0
            image_binaryHor = image_greyscaleHor
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorRWithoutMask = np.add(ArrayHorRWithoutMask, image_binaryHor)

            print(i,'done')

    ## If no mask is needed. or masking cannot be done correctly because of reflective surface set nomask to True. This way no mask will be added
    nomask = False
    if nomask == True:
        maskX = np.zeros_like(ArrayVertLWithoutMask)
        maskY= np.zeros_like(ArrayVertRWithoutMask)

    ################################################# APPLYING MASKS TO DIFFERENT MATRICES FOR LATER USE #########################################################
    arrayVertLMasked = ma.masked_array(ArrayVertLWithoutMask,mask= maskX)
    arrayVertLMasked = ma.filled(arrayVertLMasked, '0')

    arrayHorLMasked = ma.masked_array(ArrayHorLWithoutMask, mask=maskX)
    arrayHorLMasked = ma.filled(arrayHorLMasked, '0')

    arrayVertRMasked = ma.masked_array(ArrayVertRWithoutMask, mask=maskY)
    arrayVertRMasked = ma.filled(arrayVertRMasked, '0')

    arrayHorRMasked = ma.masked_array(ArrayHorRWithoutMask, mask=maskY)
    arrayHorRMasked = ma.filled(arrayHorRMasked, '0')

    end = time.time()
    totaltime = end - start
    print('Total computing time of method ', InputParameters.methodOfTriangulation, ": ", str(totaltime))

## Function to get corresponding decimal values for a binary gray code
def inversegrayCode(n):
    inv = 0
    # Taking xor until
    # n becomes zero
    while (n):
        inv = inv ^ n
        n = n >> 1
    return inv

## Function to get decimal value of gray code
def Gray2Dec():
    global arrayVertLMasked
    global arrayHorLMasked
    global arrayVertRMasked
    global arrayHorRMasked
    print('Converting Gray to dec code')
    for i in range(heightL):
        for j in range(widthL):
            arrayVertLMasked[i][j] = inversegrayCode(int(arrayVertLMasked[i][j], 2))
            arrayHorLMasked[i][j] = inversegrayCode(int(arrayHorLMasked[i][j], 2))

    for i in range(heightR):
        for j in range(widthR):
            arrayVertRMasked[i][j] = inversegrayCode(int(arrayVertRMasked[i][j], 2))
            arrayHorRMasked[i][j] = inversegrayCode(int(arrayHorRMasked[i][j], 2))

## Extra function for plotting a gradient picture of the scene.
def gradientImage():
    x = np.array(arrayVertLMasked, dtype=float)
    plt.imshow(x, cmap='jet')
    plt.show()

##Function to create inverse matrices used for easier correspondence matching. Gets used in Triangulaton()
def FindCorrespondence():

    ## Creating 3D arrays to represent horizontal AND vertical binary codes for each pixel.
    LeftCamera = np.stack((arrayVertLMasked, arrayHorLMasked), axis=2)
    RightCamera = np.stack((arrayVertRMasked, arrayHorRMasked), axis=2)

    ## Adjusting view for 3D arrays for easier matrix usage. Both arrays LeftCamera and LeftCameraPleaseWork are identical just differently shown.
    #LeftCameraPleaseWork = LeftCamera.view([(f'f{i}', LeftCamera.dtype) for i in range(LeftCamera.shape[-1])])[
     #   ..., 0].astype('O')
    #RightCameraPleaseWork = RightCamera.view([(f'f{i}', RightCamera.dtype) for i in range(RightCamera.shape[-1])])[
     #   ..., 0].astype('O')

    import time
    start = time.time()

    ## Find max values to create empty (0) matrices to use
    MaxValue1 = np.amax(arrayVertLMasked)
    MaxValue2 = np.amax(arrayHorLMasked)
    MaxValue3 = np.amax(arrayVertRMasked)
    MaxValue4 = np.amax(arrayHorRMasked)


    InversArrayRowLeft = np.zeros((MaxValue1 + 1,MaxValue2 + 1))
    InversArrayColumnLeft = np.zeros((MaxValue1 + 1,MaxValue2 + 1))
    InversArrayRowRight = np.zeros((MaxValue3 + 1, MaxValue4 + 1))
    InversArrayColumnRight = np.zeros((MaxValue3 + 1, MaxValue4 + 1))

    ##Loop to generate invers matrix of Left Camera matrix for pixel correspondence
    for i in range (heightL):
        for j in range (widthL):
            indexRowLeft = arrayVertLMasked[i][j]
            indexColumnLeft = arrayHorLMasked[i][j]
            InversArrayRowLeft[indexRowLeft][indexColumnLeft] = i
            InversArrayColumnLeft[indexRowLeft][indexColumnLeft] = j

    TempMatrixLeft = np.stack((InversArrayColumnLeft, InversArrayRowLeft), axis=2)
    InversMatrixLeftCam = TempMatrixLeft.view([(f'f{i}', TempMatrixLeft.dtype) for i in range(TempMatrixLeft.shape[-1])])[
        ..., 0].astype('O')

    ##Loop to generate invers matrix of Right Camera matrix for pixel correspondence
    for i in range (heightR):
        for j in range(widthR):
            indexRowRight = arrayVertRMasked[i][j]
            indexColumnRight = arrayHorRMasked[i][j]
            InversArrayRowRight[indexRowRight][indexColumnRight] = i
            InversArrayColumnRight[indexRowRight][indexColumnRight] = j

    TempMatrixRight = np.stack((InversArrayColumnRight, InversArrayRowRight),axis =2)
    InversMatrixRightCam = TempMatrixRight.view([(f'f{i}', TempMatrixRight.dtype) for i in range(TempMatrixRight.shape[-1])])[
        ..., 0].astype('O')

    end = time.time()
    totaltime = end - start
    print('tijd omzetten inverse matrix = ', totaltime)
    return InversMatrixLeftCam,InversMatrixRightCam

