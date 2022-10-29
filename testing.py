#@defgroup cam_python CAMERA_TOOLBOX_python
#
#These classes are part of the Op3Mech CAMERA toolbox the toolbox is
#available on bitbucket. go to : https://bitbucket.org/SeppeSels/camera_toolbox_python
#
#copyright Seppe Sels Invilab University of Antwerp 03-09-2021
#
#This code is for internal use only (Uantwerpen), please follow CC BY-NC-SA with the additional restriction
#that it cannot be shared outside of the University of Antwerp.
#Bugs, bugfixes and additions to the code need to be reported to Invilab (contact: Seppe Sels)
#for other licences, contact author.
#
#@ingroup cam_python



from CameraModel.Pleora.RGB.GenericRGBCamera import GenericRGBCamera

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    def plotimage(img, name='image'):
        plt.imshow(img.astype(np.uint8))
        plt.show()


    cam = GenericRGBCamera()

    check = cam.Open("2BA200004267")
    f = cam.GetFrame()
    cam.Close()
    plotimage(f)
    a=1
