from PyCBD.pipelines import CBDPipeline, prepare_image
from PyGeiger.detector import GeigerDetector
from PyCBD.checkerboard_detection.detectors import CheckerboardDetector
import cv2 as cv
import os
import glob as glob
import matplotlib.pyplot as plt

os.chdir("..")
print(os.getcwd())
fns = glob.glob(r"C:\Users\mheva\OneDrive\Bureaublad\Univerisity_Work\Semester-2\Visie\checkerboard_test\*")
#I = plt.imread(fns[2])
#if I.ndim == 3: im = I.mean(axis=2)

I = cv.imread(fns[2])
n = max(int(I.shape[0]/1000),1)
if n > 1:   im = I[::n,::n]
#detector = GeigerDetector()
detector = CBDPipeline(detector=GeigerDetector() ,expand=True, predict=True)
#detector = CheckerboardDetector()

detector.must_plot_iterations = True
results, board_uv, board_xy = detector.detect_checkerboard(im)
#board_uv, board_xy, corners_uv = detector.detect_checkerboard(im)
board_uv = board_uv*n
print(board_uv)
plt.imshow(I)
plt.plot(board_uv[:, 0], board_uv[:, 1], 'go', markeredgecolor='k')
for i in range(0, board_uv.shape[0]):
    plt.text(board_uv[i, 0], board_uv[i, 1], i, color="black")
title = "Detected checkerboard"
plt.title(title)
plt.show()