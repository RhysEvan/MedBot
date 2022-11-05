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
try:
    from connections.cameras import *
    pleora_lib = True
except:
    print("Pleora Library not Installed")
    pleora_lib = False



class app_stitching(QMainWindow, Ui_MainWindow):
    def __init__(self, port = None, pleora= True):
        super().__init__()
        self.robot = "HangingArm"
        self.vis_path = True
        self.animate = False
        ## Initialisation of GUI ##
        ## if changes are made to GUI then please add chosen_bot callable to the setupUi function and to self.visual = interface(self.centralwidget, robot=chosen_bot)
        self.setupUi(self,choice = self.robot)
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

        ## call to button functions and their forward to internal functions ##
        self.homing.clicked.connect(self.var.main_home)
        self.submit.clicked.connect(self.var.append_coord)
        self.recording.clicked.connect(self.var.append_motor)
        self.compiling.clicked.connect(self.var.json_file)
        self.remove_motor.clicked.connect(self.var.handle_motorlist)
        self.executing.clicked.connect(self.var.run_json)
        self.animating.clicked.connect(self.var.animation_seq)
        self.path.clicked.connect(self.var.dynamic.visible_path)
        self.robot_options.clicked.connect(self.var.dynamic.clicked)

        self.aabs.valueChanged.connect(self.var.dynamic.slider_input)
        self.babs.valueChanged.connect(self.var.dynamic.slider_input)
        self.cabs.valueChanged.connect(self.var.dynamic.slider_input)
        self.dabs.valueChanged.connect(self.var.dynamic.slider_input)
        self.eabs.valueChanged.connect(self.var.dynamic.slider_input)
        self.fabs.valueChanged.connect(self.var.dynamic.slider_input)
        self.xcoord.textEdited.connect(self.x_location)
        self.ycoord.textEdited.connect(self.y_location)
        self.zcoord.textEdited.connect(self.z_location)
        self.alfacoord.textEdited.connect(self.alfa_location)
        self.betacoord.textEdited.connect(self.beta_location)
        self.gammacoord.textEdited.connect(self.gamma_location)
 
        ########################
        self.show()
        ###########################

    ########################################

    def keyPressEvent(self, e):
        print(e.key())
        #every key has a digit value press any key when screen is open to see value
        #when entered into a label enter needs to be pressed 
        delete = 16777216 #esc to undo
        enter = 16777220 #enter to execute
        if e.key() == delete:
            print("removing last coordinate")
            self.var.handle_coordlist()

        if e.key() == enter:
            print("starting execute of absolute coordinates.")
            self.var.query()

    ################################################################################
    ####################### absolute movement ######################################
    ################################################################################

    def x_location(self):
        self.x_loc = self.xcoord.text()
        print(self.x_loc)

    def y_location(self):
        self.y_loc = self.ycoord.text()
        print(self.y_loc)

    def z_location(self):
        self.z_loc = self.zcoord.text()
        print(self.z_loc)

    def alfa_location(self):
        self.alfa_loc = self.alfacoord.text()
        print(self.alfa_loc)

    def beta_location(self):
        self.beta_loc = self.betacoord.text()
        print(self.beta_loc)

    def gamma_location(self):
        self.gamma_loc = self.gammacoord.text()
        print(self.gamma_loc)

    ################################################################################
    ####################### threading method  ######################################
    ################################################################################

    def image_update_left(self, Image):
        self.camera_left.setPixmap(QPixmap.fromImage(Image))
    
    def image_update_right(self, Image):
        self.camera_right.setPixmap(QPixmap.fromImage(Image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = app_stitching(pleora=pleora_lib)

    sys.exit(app.exec())
