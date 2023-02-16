#!/usr/bin/python
'''
 separate_direct_global.py
 Copyright(c) Photometry Group in Prof. Yagi's lab., ISIR, Osaka University.
     Sep. 2013, Kenichiro Tanaka
 Separate into direct and global components.

 Notes
 -----
 1. All images in the directory are loaded for separation.
 2. Black bias is automatically enabled if there is 'black.png'.
 3. Normal illumination is automatically ignored if there is 'white.png'.
'''
import os

import Graycode
import InputParameters



def separateGlobalDirect(dir):
    import argparse
    import glob
    import sys
    import numpy as np
    import cv2

    os.chdir(dir)

    # Parse program arguments
    parser = argparse.ArgumentParser(description='Separate into direct and global components.')
    parser.add_argument('-v', dest='max_min_output', default=False, action='store_true', help="Outputs max and min images.")
    parser.add_argument('-e', '--extension', default=".png", help="File extension of all images. default is .png")
    parser.add_argument('-d', '--dir', default="./", help="Source images' directory. default is the current directory.");
    parser.add_argument('-w', '--whiteout', default=False, action='store_true',
                        help="Processing mode of saturated pixel. If this flag is specified, direct component becomes white, otherwise becomes black.")
    args = parser.parse_args()

    # Variables
    black_bias = False

    # Get input filenames.
    dir_name = args.dir
    extension = args.extension
    if not dir_name.endswith('/'):
        dir_name = dir_name + '/'
    if not extension.startswith('.'):
        extension = '.' + extension
    search_sequence = dir_name + "*" + extension
    black_file = dir_name + "black" + extension
    files = glob.glob(search_sequence)
    if black_file in files:
        black_bias = True
        files.remove(black_file)
    for excp in ['white', 'direct', 'global', 'max', 'min']:
        filename = dir_name + excp + extension
        if filename in files:
            files.remove(filename)

    # If file does not exist, exit the program.
    if len(files) == 0:
        print("No images...")
        sys.exit()

    # Load images
    img = cv2.imread(files[0], -1)
    max_img = img
    min_img = img
    for filename in files:
        img = cv2.imread(filename, -1)
        max_img = np.maximum(max_img, img)
        min_img = np.minimum(min_img, img)

    img_is_16bit = (max_img.itemsize != 1)

    # If all images are satulated, direct image should be white?
    if args.whiteout:
        if img_is_16bit:
            min_img[min_img == 65535] = 0
        else:
            min_img[min_img == 255] = 0

    # Separate into direct and global components
    if black_bias:
        # subtract black bias with underflow prevention
        black_img = cv2.imread(black_file, -1)
        max_img = np.maximum(max_img - black_img, 0)
        min_img = np.maximum(min_img - black_img, 0)
    direct_img = max_img - min_img

    # Prevent overflow
    intensity_max = 65535.0 if img_is_16bit else 255.
    global_img = np.minimum(2.0 * min_img, intensity_max)
    if img_is_16bit:
        global_img = np.uint16(global_img)
    else:
        global_img = np.uint8(global_img)

    # Save images
    cv2.imwrite(dir_name + "direct" + extension, direct_img)
    cv2.imwrite(dir_name + "global" + extension, global_img)
    if args.max_min_output:
        cv2.imwrite(dir_name + 'max' + extension, max_img)
        cv2.imwrite(dir_name + 'min' + extension, min_img)



def RobustPixelThresh(threshold,binaryMaxValue,e):
    import cv2 as cv

    for i in range(Graycode.length-1):
        os.chdir(InputParameters.CalibratedImageDirectory)
        imageVert = cv.imread('imgVertCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
        image_greyscaleVert = cv.cvtColor(imageVert, cv.COLOR_BGR2GRAY)
        ret, image_binaryVert = cv.threshold(image_greyscaleVert, threshold, binaryMaxValue, cv.THRESH_BINARY)
        imageVertINV = cv.imread('imgVertINVCAML{}.png'.format(i), cv.IMREAD_UNCHANGED)
        image_greyscaleVertINV = cv.cvtColor(imageVertINV, cv.COLOR_BGR2GRAY)

        os.chdir(InputParameters.WorkingDirectory)
        imgGlobal = cv.imread('global.png')
        imgGlobalGray = cv.cvtColor(imgGlobal, cv.COLOR_BGR2GRAY)
        imgDirect =cv.imread('direct.png')
        imgDirectGray = cv.cvtColor(imgDirect, cv.COLOR_BGR2GRAY)
        for ii in range(400):
            for jj in range(800):
                if imgDirectGray[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleVert[ii][jj] > image_greyscaleVertINV[ii][jj] + e:
                    image_binaryVert[ii][jj] = 1
                elif imgDirectGray[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleVert[ii][jj] + e < image_greyscaleVertINV[ii][jj]:
                    image_binaryVert[ii][jj] = 0
                elif image_greyscaleVert[ii][jj] + e < imgDirectGray[ii][jj] and image_greyscaleVertINV[ii][jj] > imgGlobalGray[ii][jj] + e:
                    image_binaryVert[ii][jj] = 0
                elif image_greyscaleVert[ii][jj] > imgGlobalGray[ii][jj] + e and image_greyscaleVertINV[ii][jj] + e > imgDirectGray[ii][jj]:
                    image_binaryVert[ii][jj] = 1

        image_binaryVert[image_binaryVert > 0] = 255
        if i == 6 or 7 or 8 or 5:
            cv.imwrite("testMeanThreshCAML{}.png".format(i), image_binaryVert)

