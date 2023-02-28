import os
import cv2 as cv
import numpy as np
import math
import sys
import numpy.matlib
from PIL import Image
from .Camera1 import *
from .InputParameters import *
np.set_printoptions(threshold=sys.maxsize)

N = math.ceil(math.log2(width))
offset = math.floor((2 ^ N - 1) / 2)
lengte = np.binary_repr((2 ** N) - 1)
length = len(np.binary_repr((2 ** N) - 1))
getal = []
matrix = []
arr = []
pyint = []
c = []


class GrayCode():
    def __init__(self):
        pass

    def BinaireCode(self):
        global getal
        global arr
        global c
        for i in range(2 ** N):
            number = np.binary_repr(i, width=length)

            for i in range(length):
                getal = np.append(getal, int(number[i]))

        arr = np.reshape(getal, (2 ** N, length))
        arr = arr.astype((int))
        # print(arr)
        # print('arr kolom 1 = ', arr[:, 0])
        c = arr[:, 0]

    def GrayCode(self):
        ## Eerste kolom kopieren
        global matrix
        global pyint
        for i in range(2 ** N):
            bit = c[i]
            matrix.append(bit)

        for i in range(1, length):
            a = arr[:, i - 1]  # eerste kolom
            b = arr[:, i]  # tweede kolom
            # print(a)
            # print(b)
            for i in range(2 ** N):
                temp = np.bitwise_xor(bool(a[i].all()), bool(b[i].all()))
                matrix.append(temp)
        # print(arr)
        matrix = np.reshape(matrix, (length, 2 ** N))

        pyint = getattr(matrix, "tolist", lambda: matrix)()  ## Van numpy.int32 omvormen naar python int

        # print('array pyint, graycode array van alle kolommen  = ', pyint)
        # print(type(pyint[1]))

    def RepmatVertical(self):
        for i in range(length):
            row = np.array(pyint[i])
            row2 = np.matlib.repmat(row, 1, height)
            row3 = np.reshape(row2, (height, 2 ** N))
            row4 = np.array(row3, dtype=bool)

            image1 = Image.fromarray(row4)
            # image.show()
            image2 = image1.resize(image_resolution)    # resize om de juiste grootte te hebben voor full screen (zie Projector.monitor_size)
            os.chdir(InputParameters.ImageDirectory)                                     ## Temporary way of saving images in certain file (still needs to be changed)
            image2.save('graycodeVert{}.png'.format(i))
            ## Saving invers patterns
            grayCode = cv.imread('graycodeVert{}.png'.format(i))
            ret, imageGrayInv = cv.threshold(grayCode, 127, 255, cv.THRESH_BINARY_INV)
            cv.imwrite("graycodeVertInv{}.png".format(i), imageGrayInv)
            os.chdir(InputParameters.WorkingDirectory)

    def RepmatHorizontal(self):
        for i in range(length):
            tester = matrix[i]
            column = np.reshape(tester, (2 ** N, 1))

            column2 = np.matlib.repmat(column, 1, width)

            column3 = np.array(column2, dtype=bool)
            image2 = Image.fromarray(column3)
            image3 = image2.resize(image_resolution)
            os.chdir(InputParameters.ImageDirectory)
            image3.save('graycodeHor{}.png'.format(i))
            ## Saving invers patterns
            grayCode = cv.imread('graycodeHor{}.png'.format(i))
            ret, imageGrayInv = cv.threshold(grayCode, 127, 255, cv.THRESH_BINARY_INV)
            cv.imwrite("graycodeHorInv{}.png".format(i), imageGrayInv)
            os.chdir(InputParameters.WorkingDirectory)
    def ImageResize(self):

        pass





