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
    ## 1) Simple method using simple thresholding
    if methodParameter == 1:
        #threshold = input('Threshold ? : ')
        #threshold = int(threshold)
        ArrayVertLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayHorLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayVertRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        ArrayHorRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)

        for i in range(Graycode.length-numberOfImages):          ## -1 because resolution of projector is higher than camera's
            os.chdir(InputParameters.CalibratedImageDirectory)

            ## array for Vertical Patterns
            imageVert = cv.imread('imgVertCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
            if i == 6 or 7 or 8 or 5:
                cv.imwrite("testSimpleCAML{}.png".format(i), imagebinarytest)
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertLWithoutMask = np.add(ArrayVertLWithoutMask, image_binaryVert)

            ## Horizontal patterns
            imageHor = cv.imread('imgHorCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorLWithoutMask = np.add(ArrayHorLWithoutMask, image_binaryHor)

            ## Vertical patterns
            imageVert = cv.imread('imgVertCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertRWithoutMask = np.add(ArrayVertRWithoutMask, image_binaryVert)

            ## Horizontal patterns
            imageHor = cv.imread('imgHorCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorRWithoutMask = np.add(ArrayHorRWithoutMask, image_binaryHor)
            print(i, 'Done')

    ## 10) Simple method using simple thresholding (with adjustable interactive threshold)
    if methodParameter == 10:
        ArrayVertLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayHorLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayVertRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        ArrayHorRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        def on_changeThresh(value):
            global getThresh
            ##Read white image Left camera
            imageAllWhite = cv.imread('imgVertCAML7.png', cv.IMREAD_UNCHANGED)
            greyscaleAllWhite = cv.cvtColor(imageAllWhite, cv.COLOR_BGR2GRAY)
            imgWhiteL = cv.threshold(greyscaleAllWhite, value, 255, cv.THRESH_BINARY)
            imgWhiteL = np.asarray(imgWhiteL[1])
            ##Read black image Left camera
            imageAllBlack = cv.imread('imgVertCAML8.png', cv.IMREAD_UNCHANGED)
            greyscaleAllBlack = cv.cvtColor(imageAllBlack, cv.COLOR_BGR2GRAY)
            imgBlackL = cv.threshold(greyscaleAllBlack, value, 255, cv.THRESH_BINARY)
            imgBlackL = np.asarray(imgBlackL[1])

            ##stacking to show all binary images in one window simultaneously
            img = np.hstack((imgWhiteL, imgBlackL))
            cv.imshow('threshold', img)
            ## Making threshold a function variable to acces later
            on_changeThresh.outputThreshold = int(cv.getTrackbarPos('Threshold', 'threshold'))

        print('Please adjust threshold parameter thresholding. When ready press any key to quit.')
        cv.namedWindow('threshold')
        cv.createTrackbar('Threshold', 'threshold', 0, 255, on_changeThresh)
        cv.waitKey(0)
        cv.destroyAllWindows()
        threshold = on_changeThresh.outputThreshold
        print('Threshold = ', threshold)

        for i in range(Graycode.length-numberOfImages):          ## -1 because resolution of projector is higher than camera's
            os.chdir(InputParameters.CalibratedImageDirectory)

            ## array for Vertical Patterns
            imageVert = cv.imread('imgVertCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
            imagebinarytest = image_binaryVert.copy()  ## (Tijdelijk om te zien of zwart wit afbeelding in orde is)
            imagebinarytest[imagebinarytest > 0] = 255
            if i == 6 or 7 or 8 or 5:
                cv.imwrite("testSimpleAdjustedCAML{}.png".format(i), imagebinarytest)
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertLWithoutMask = np.add(ArrayVertLWithoutMask, image_binaryVert)

            ## Horizontal patterns
            imageHor = cv.imread('imgHorCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorLWithoutMask = np.add(ArrayHorLWithoutMask, image_binaryHor)

            ## Vertical patterns
            imageVert = cv.imread('imgVertCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertRWithoutMask = np.add(ArrayVertRWithoutMask, image_binaryVert)

            ## Horizontal patterns
            imageHor = cv.imread('imgHorCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorRWithoutMask = np.add(ArrayHorRWithoutMask, image_binaryHor)
            print(i, 'Done')

    ## 2) Combining adaptive with simple threshold
    if methodParameter == 2:
        ArrayVertLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayHorLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayVertRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        ArrayHorRWithoutMask= np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)

        threshold = input('Threshold ? : ')
        threshold = int(threshold)
        for i in range(Graycode.length-numberOfImages):
            os.chdir(InputParameters.CalibratedImageDirectory)

            ## array for Vertical Patterns
            imageVert = cv.imread('imgVertCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            if i <= 6:
                ret,image_binaryVert = cv.threshold(image_greyscaleVert,threshold,binaryMaxValue,cv.THRESH_BINARY)
            else:
                image_binaryVert = cv.adaptiveThreshold(image_greyscaleVert, binaryMaxValue, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,3,1)
            imagebinarytest = image_binaryVert.copy()  ## (Tijdelijk om te zien of zwart wit afbeelding in orde is)
            imagebinarytest[imagebinarytest > 0] = 255
            if i == 6 or 7 or 8 or 5:
                cv.imwrite("testCombinedCAML{}.png".format(i), imagebinarytest)
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertLWithoutMask = np.add(ArrayVertLWithoutMask,image_binaryVert)

            ## Horizontal patterns
            imageHor = cv.imread('imgHorCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            if i <= 6:
                ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
            else:
                image_binaryHor = cv.adaptiveThreshold(image_greyscaleHor,binaryMaxValue,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,3,1)
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorLWithoutMask = np.add(ArrayHorLWithoutMask, image_binaryHor)

            ## Vertical patterns
            imageVert = cv.imread('imgVertCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            if i <= 6:
                ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
            else:
                image_binaryVert = cv.adaptiveThreshold(image_greyscaleVert, binaryMaxValue, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3,1)
            if i == 6 or 7 or 8 or 5:                           ## (Tijdelijk om te zien of zwart wit afbeelding in orde is)
                cv.imwrite("testThreshCAMR{}.png".format(i), image_binaryVert)
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertRWithoutMask = np.add(ArrayVertRWithoutMask, image_binaryVert)

            ## Horizontal patterns
            imageHor = cv.imread('imgHorCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            if i <= 6:
                ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
            else:
                image_binaryHor = cv.adaptiveThreshold(image_greyscaleHor, binaryMaxValue, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 1)
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorRWithoutMask = np.add(ArrayHorRWithoutMask, image_binaryHor)
            print(i,'Done')

    ## 3) Using adaptive threshold
    if methodParameter == 3:
        ArrayVertLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayHorLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayVertRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        ArrayHorRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        for i in range(Graycode.length-numberOfImages):
            os.chdir(InputParameters.CalibratedImageDirectory)

            ## array for Vertical Patterns
            imageVert = cv.imread('imgVertCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            image_binaryVert = cv.adaptiveThreshold(image_greyscaleVert, binaryMaxValue, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 45, 1)
            imagebinarytest = image_binaryVert.copy()  ## (Tijdelijk om te zien of zwart wit afbeelding in orde is)
            imagebinarytest[imagebinarytest > 0] = 255
            if i == 6 or 7 or 8 or 5:
                cv.imwrite("testAdaptiveCAML{}.png".format(i), imagebinarytest)
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertLWithoutMask = np.add(ArrayVertLWithoutMask, image_binaryVert)

            ## Horizontal patterns
            imageHor = cv.imread('imgHorCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            image_binaryHor = cv.adaptiveThreshold(image_greyscaleHor, binaryMaxValue, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 45, 1)
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorLWithoutMask = np.add(ArrayHorLWithoutMask, image_binaryHor)

            ## Vertical patterns
            imageVert = cv.imread('imgVertCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            image_binaryVert = cv.adaptiveThreshold(image_greyscaleVert, binaryMaxValue, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 45, 1)
            if i == 6 or 7 or 8 or 5:  ## (Tijdelijk om te zien of zwart wit afbeelding in orde is)
                cv.imwrite("testMeanThreshCAMR{}.png".format(i), image_binaryVert)
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertRWithoutMask = np.add(ArrayVertRWithoutMask, image_binaryVert)

            ## Horizontal patterns
            imageHor = cv.imread('imgHorCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            image_binaryHor = cv.adaptiveThreshold(image_greyscaleHor, binaryMaxValue, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 45, 1)
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorRWithoutMask = np.add(ArrayHorRWithoutMask, image_binaryHor)
            print(i, 'Done')

    ## 4) Using per pixel threshold from mean of allWhite and allBlack image
    if methodParameter == 4:
        a,b = findingThresholdShadowMask()
        ArrayVertLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayHorLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayVertRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        ArrayHorRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        for i in range(Graycode.length-numberOfImages):
                os.chdir(InputParameters.CalibratedImageDirectory)

                ## array for Vertical Patterns
                imageVert = cv.imread('imgVertCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
                image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
                ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
                image_binaryVert = image_greyscaleVert < ThresholdL
                image_binaryVert = image_binaryVert.astype(int)
                imagebinarytest = image_binaryVert.copy()           ## (Tijdelijk om te zien of zwart wit afbeelding in orde is)
                imagebinarytest[imagebinarytest > 0] = 255
                if i == 6 or 7 or 8 or 5:
                    cv.imwrite("testaverageBlackWhiteCAML{}.png".format(i), imagebinarytest)
                image_binaryVert = image_binaryVert.astype('str')
                ArrayVertLWithoutMask = np.add(ArrayVertLWithoutMask, image_binaryVert)

                ## Horizontal patterns
                imageHor = cv.imread('imgHorCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
                image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
                ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
                image_binaryHor = image_greyscaleHor < ThresholdL
                image_binaryHor = image_binaryHor.astype(int)
                image_binaryHor = image_binaryHor.astype('str')
                ArrayHorLWithoutMask = np.add(ArrayHorLWithoutMask, image_binaryHor)

                ## Vertical patterns
                imageVert = cv.imread('imgVertCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
                image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
                ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
                image_binaryVert = image_greyscaleVert < ThresholdR
                image_binaryVert = image_binaryVert.astype(int)
                if i == 6 or 7 or 8 or 5 or 9:  ## (Tijdelijk om te zien of zwart wit afbeelding in orde is)
                    cv.imwrite("testMeanThreshCAMR{}.png".format(i), image_binaryVert)
                    cv.imwrite("testGreyscale{}.png".format(i),image_greyscaleVert)
                image_binaryVert = image_binaryVert.astype('str')
                ArrayVertRWithoutMask = np.add(ArrayVertRWithoutMask, image_binaryVert)

                ## Horizontal patterns
                imageHor = cv.imread('imgHorCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
                image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
                ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
                image_binaryHor = image_greyscaleHor < ThresholdR
                image_binaryHor = image_binaryHor.astype(int)
                image_binaryHor = image_binaryHor.astype('str')
                ArrayHorRWithoutMask = np.add(ArrayHorRWithoutMask, image_binaryHor)
                print(i, 'Done')

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

    ## 6) Robust Pixel Classification
    if methodParameter == 6:
        import shutil
        import GlobalDirect

        e = 0
        ArrayVertLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayHorLWithoutMask = np.full((imgL.shape[0], imgL.shape[1]), "", dtype=object)
        ArrayVertRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)
        ArrayHorRWithoutMask = np.full((imgR.shape[0], imgR.shape[1]), "", dtype=object)

        ### Copying images in their own folder
        for i in range(5, 9):
            original = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\imgVertCAML{}.png".format(i)
            newLocation = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesLeft\imgVertCAML{}.png".format(i)
            shutil.copyfile(original, newLocation)
            original = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\imgVertINVCAML{}.png".format(i)
            newLocation = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesLeft\imgVertINVCAML{}.png".format(i)
            shutil.copyfile(original, newLocation)
            original = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\imgHorCAML{}.png".format(i)
            newLocation = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesLeft\imgHorCAML{}.png".format(i)
            shutil.copyfile(original, newLocation)
            original = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\imgHorINVCAML{}.png".format(i)
            newLocation = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesLeft\imgHorINVCAML{}.png".format(i)
            shutil.copyfile(original, newLocation)
            original = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\ThresholdCAML0.png"
            newLocation = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesLeft\white.png"
            shutil.copyfile(original, newLocation)
            original = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\ThresholdCAML1.png"
            newLocation = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesLeft\black.png"
            shutil.copyfile(original, newLocation)

            original2 = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\imgVertCAMR{}.png".format(i)
            newLocation2 = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesRight\imgVertCAMR{}.png".format(i)
            shutil.copyfile(original2, newLocation2)
            original2 = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\imgVertINVCAMR{}.png".format(i)
            newLocation2 = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesRight\imgVertINVCAMR{}.png".format(i)
            shutil.copyfile(original2, newLocation2)
            original2 = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\imgHorCAMR{}.png".format(i)
            newLocation2 = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesRight\imgHorCAMR{}.png".format(i)
            shutil.copyfile(original2, newLocation2)
            original2 = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\imgHorINVCAMR{}.png".format(i)
            newLocation2 = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesRight\imgHorINVCAMR{}.png".format(i)
            shutil.copyfile(original2, newLocation2)
            original2 = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\ThresholdCAMR0.png"
            newLocation2 = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesRight\white.png"
            shutil.copyfile(original2, newLocation2)
            original2 = r"C:\Users\Diane\PycharmProjects\pythonProject\CalibratedImages\ThresholdCAMR1.png"
            newLocation2 = r"C:\Users\Diane\PycharmProjects\pythonProject\imagesRight\black.png"
            shutil.copyfile(original2, newLocation2)

        try :
                os.remove(InputParameters.imagesLeft + '\global.png')
                os.remove(InputParameters.imagesLeft + '\direct.png')
                os.remove(InputParameters.imagesRight + '\global.png')
                os.remove(InputParameters.imagesRight + '\direct.png')
        except:
            pass

        GlobalDirect.separateGlobalDirect(InputParameters.imagesLeft)
        GlobalDirect.separateGlobalDirect(InputParameters.imagesRight)


        for i in range(Graycode.length - numberOfImages):

            os.chdir(InputParameters.imagesLeft)
            imgGlobal = cv.imread('global.png')
            imgGlobalGray = cv.cvtColor(imgGlobal, cv.COLOR_BGR2GRAY)
            imgDirect = cv.imread('direct.png')
            imgDirectGray = cv.cvtColor(imgDirect, cv.COLOR_BGR2GRAY)

            os.chdir(InputParameters.CalibratedImageDirectory)
            imageVert = cv.imread('imgVertCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
            imageVertINV = cv.imread('imgVertINVCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVertINV = cv.cvtColor(imageVertINV, cv.COLOR_BGR2GRAY)
            for ii in range(image_greyscaleVert.shape[0]):
                for jj in range(image_greyscaleVert.shape[1]):
                    if imgDirectGray[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleVert[ii][jj] > \
                            image_greyscaleVertINV[ii][jj] + e:
                        image_binaryVert[ii][jj] = 1
                    elif imgDirectGray[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleVert[ii][jj] + e < \
                            image_greyscaleVertINV[ii][jj]:
                        image_binaryVert[ii][jj] = 0
                    elif image_greyscaleVert[ii][jj] + e < imgDirectGray[ii][jj] and image_greyscaleVertINV[ii][jj] > \
                            imgGlobalGray[ii][jj] + e:
                        image_binaryVert[ii][jj] = 0
                    elif image_greyscaleVert[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleVertINV[ii][
                        jj] + e < imgDirectGray[ii][jj]:
                        image_binaryVert[ii][jj] = 1
            image_binartVertcopy = image_binaryVert.copy()
            image_binartVertcopy[image_binartVertcopy > 0] = 255
            if i == 6 or 7 or 8 or 5 or 9:
                cv.imwrite("testRobustPixelThreshCAML{}.png".format(i), image_binartVertcopy)

            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertLWithoutMask = np.add(ArrayVertLWithoutMask, image_binaryVert)



            imageHor = cv.imread('imgHorCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
            imageHorINV = cv.imread('imgHorINVCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHorINV = cv.cvtColor(imageHorINV, cv.COLOR_BGR2GRAY)
            for ii in range(image_greyscaleHor.shape[0]):
                for jj in range(image_greyscaleHor.shape[1]):
                    if imgDirectGray[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleHor[ii][jj] > \
                            image_greyscaleHorINV[ii][jj] + e:
                        image_binaryHor[ii][jj] = 1
                    elif imgDirectGray[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleHor[ii][jj] + e < \
                            image_greyscaleHorINV[ii][jj]:
                        image_binaryHor[ii][jj] = 0
                    elif image_greyscaleHor[ii][jj] + e < imgDirectGray[ii][jj] and image_greyscaleHorINV[ii][jj] > \
                            imgGlobalGray[ii][jj] + e:
                        image_binaryHor[ii][jj] = 0
                    elif image_greyscaleHor[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleHorINV[ii][
                        jj] + e < imgDirectGray[ii][jj]:
                        image_binaryHor[ii][jj] = 1
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorLWithoutMask = np.add(ArrayHorLWithoutMask, image_binaryHor)


            os.chdir(InputParameters.imagesRight)
            imgGlobal = cv.imread('global.png')
            imgGlobalGray = cv.cvtColor(imgGlobal, cv.COLOR_BGR2GRAY)
            imgDirect = cv.imread('direct.png')
            imgDirectGray = cv.cvtColor(imgDirect, cv.COLOR_BGR2GRAY)

            os.chdir(InputParameters.CalibratedImageDirectory)
            imageVert = cv.imread('imgVertCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
            ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
            imageVertINV = cv.imread('imgVertINVCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleVertINV = cv.cvtColor(imageVertINV, cv.COLOR_BGR2GRAY)
            for ii in range(image_greyscaleVert.shape[0]):
                for jj in range(image_greyscaleVert.shape[1]):
                    if imgDirectGray[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleVert[ii][jj] > \
                            image_greyscaleVertINV[ii][jj] + e:
                        image_binaryVert[ii][jj] = 1
                    elif imgDirectGray[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleVert[ii][jj] + e < \
                            image_greyscaleVertINV[ii][jj]:
                        image_binaryVert[ii][jj] = 0
                    elif image_greyscaleVert[ii][jj] + e < imgDirectGray[ii][jj] and image_greyscaleVertINV[ii][jj] > \
                            imgGlobalGray[ii][jj] + e:
                        image_binaryVert[ii][jj] = 0
                    elif image_greyscaleVert[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleVertINV[ii][
                        jj] + e < imgDirectGray[ii][jj]:
                        image_binaryVert[ii][jj] = 1
            image_binaryVert = image_binaryVert.astype('str')
            ArrayVertRWithoutMask = np.add(ArrayVertRWithoutMask, image_binaryVert)

            imageHor = cv.imread('imgHorCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHor = cv.cvtColor(imageHor, cv.COLOR_BGR2GRAY)
            ret, image_binaryHor = cv.threshold(image_greyscaleHor, threshold, binaryMaxValue, cv.THRESH_BINARY)
            imageHorINV = cv.imread('imgHorINVCAMR{}.png'.format(i), cv.IMREAD_UNCHANGED)
            image_greyscaleHorINV = cv.cvtColor(imageHorINV, cv.COLOR_BGR2GRAY)
            for ii in range(image_greyscaleHor.shape[0]):
                for jj in range(image_greyscaleHor.shape[1]):
                    if imgDirectGray[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleHor[ii][jj] > \
                            image_greyscaleHorINV[ii][jj] + e:
                        image_binaryHor[ii][jj] = 1
                    elif imgDirectGray[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleHor[ii][jj] + e < \
                            image_greyscaleHorINV[ii][jj]:
                        image_binaryHor[ii][jj] = 0
                    elif image_greyscaleHor[ii][jj] + e < imgDirectGray[ii][jj] and image_greyscaleHorINV[ii][jj] > \
                            imgGlobalGray[ii][jj] + e:
                        image_binaryHor[ii][jj] = 0
                    elif image_greyscaleHor[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleHorINV[ii][
                        jj] + e < imgDirectGray[ii][jj]:
                        image_binaryHor[ii][jj] = 1
            image_binaryHor = image_binaryHor.astype('str')
            ArrayHorRWithoutMask = np.add(ArrayHorRWithoutMask, image_binaryHor)
            print(i+1,' done')

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

