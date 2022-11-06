#libraries
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np 

#file based imports
from static.App_GUI import Ui_MainWindow
from connections.serial_com import *

from static.json_compiler import *
from kinematics import Kinematics
from static.backend_GUI import backend
from static.triggers import trigger
try:
    from connections.cameras import *
    pleora_lib = True
except:
    print("Pleora Library not Installed")
    pleora_lib = False



class app_stitching(QMainWindow, Ui_MainWindow):
    def __init__(self, port = None, pleora= True):
        super().__init__()
        robot = "HangingArm"
        self.vis_path = False
        self.animate = False
        ## Initialisation of GUI ##
        ## if changes are made to GUI then please add chosen_bot callable to the setupUi function and to self.visual = interface(self.centralwidget, robot=chosen_bot)
        self.setupUi(self,choice=robot)
        if pleora:
            ## Threaded Camera Left## 
            self.cam_l = Feed("2BA200004267") ## Number represents the camera adress on the computer ##

            self.cam_l.start()
            self.cam_l.ImageUpdate.connect(self.image_update_left)
        
            ## Threaded Camera Right## 
            self.cam_r = Feed("2BA200004267") ## Number represents the camera adress on the computer ##

            self.cam_r.start()
            self.cam_r.ImageUpdate.connect(self.image_update_right)
        ## conection to interface to create matplot visual ##
        self.graph = self.visual
        self.kinematics = self.graph.kin
        ## connection to arduino GRBL ##
        self.com = serial_bridge()
        ## json compiler initiation ##
        self.file = json_handler()
        ## all background related functions ##
        self.var = backend(self)
        ## build the model in the interface and create the motor list and coordinate list ##
        self.var.model_build()
        trigger(self)
        ## call to button functions and their forward to internal functions ##
        
        self.show()

    ########################################

    def keyPressEvent(self, e):
        print(e.key())
        #every key has a digit value press any key when screen is open to see value
        #when entered into a label enter needs to be pressed 
        delete = 16777216 #esc to undo
        enter = 16777220 #enter to execute
        if e.key() == enter:
            print("starting execute of absolute coordinates.")
            self.var.query()

    ####################### threading method  ######################################

    def image_update_left(self, Image):
        self.camera_left.setPixmap(QPixmap.fromImage(Image))
    
    def image_update_right(self, Image):
        self.camera_right.setPixmap(QPixmap.fromImage(Image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = app_stitching(pleora=pleora_lib)

    sys.exit(app.exec())
