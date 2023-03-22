#libraries
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np

#file based imports
from static.Robot_Control_Panel import Ui_MainWindow
from connections.serial_com_GRBL import *
from connections.serial_com_CUSTOM import *

from static.json_compiler import *
from static.backend_GUI import backend
from static.triggers import trigger
#from static.map_sequencing import Map
from connections.cameras import *
pleora_lib = True
#except:
#    print("Pleora Library not Installed")
#    pleora_lib = False

class app_stitching(QMainWindow, Ui_MainWindow):
    def __init__(self, port = None, pleora= True):
        super().__init__()
        self.vis_path = True
        self.animate = False
        self.map_run = False
        self.setupUi(self)
        ## Initialisation of GUI ##
        self.visual.kin.model_param("HangingArm")
        if pleora:
            ## Threaded Camera Left## 
            self.cam_l = Feed("2BA200004266") ## Number represents the camera adress on the computer ##

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
        self.com = serial_bridge_GRBL()
        custom = False
        if self.com.device == None:
            self.com = serial_bridge_CUSTOM()
            custom = True
        ## json compiler initiation ##
        self.file = json_handler()
        ## all background related functions ##
        self.back = backend(self,custom_check = custom)
        ##link to 3D mapping code ##
        #self.mapping = Map(self)

        #TODO IMPLEMENT THREAD WITH CORRECT TIME LOOP THAT INNITATES 3D-mapping
        ## QTimer methode ##

        #self.sequence = QTimer(self)
        #self.sequence.timeout.connect(self.process)
        #self.sequence.start(100)

        ## build the model in the interface and create the motor list and coordinate list ##
        self.back.model_build()
        ## all GUI variable function connections ##
        trigger(self)
        print("add functionality to switch between active RGB and active Mapping")
        ## call to button functions and their forward to internal functions ##       
        self.show()

    ########################################

    def keyPressEvent(self, e):

        
        #every key has a digit value press any key when screen is open to see value
        #when entered into a label enter needs to be pressed 
        delete = 16777216 #esc to undo
        enter = 16777220 #enter to execute
        if e.key() == enter:
            print("starting execute of absolute coordinates.")
            self.back.query()

    ####################### threading method  ######################################

    def image_update_left(self, img):
        if not self.map_run:
            self.camera_left.setPixmap(QPixmap.fromImage(img))
        else:
            self.mapping.handler.emit_camL = img
    
    def image_update_right(self, img):
        if not self.map_run:
            self.camera_right.setPixmap(QPixmap.fromImage(img))
        else:
            self.mapping.handler.emit_camR = img

    ##################### Movement Sequencing ##########################

    def process(self):
        self.mapping.continuous_mapping()
        # still needs a visualizer and other things like the projection of live images onto the map or the raw image onto the map
        # implement a check system that looks for the points to stitch
        # if there is a stitch initiate request for coordinates and run through inverse AI
        # send commands through serial command
        # wait till movements are ready/done
        # run predefined path for end effector


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = app_stitching(pleora=pleora_lib)
    sys.exit(app.exec())