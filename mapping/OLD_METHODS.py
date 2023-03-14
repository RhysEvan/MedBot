# all of the if statements made below came from a previous thesis, they are not used in the robot
# they are kept in for if someone wants to reimplement them, be wary that this code has not been refiteted to the current system

# methodParameter was the old way of determining which method would be used, since only one method is currently in use this variable
# has been removed. It can easily be added back.
methodParameter = 5 #this was the identifier of the currently used method when all options were listed in the Detection.py file

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
