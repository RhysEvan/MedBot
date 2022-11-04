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
        self.animating.clicked.connect(self.var.initiate_animate)
        self.path.clicked.connect(self.var.dynamic.visible_path)
        self.robot_options.clicked.connect(self.var.dynamic.clicked)

        self.aabs.valueChanged.connect(self.joint_a)
        self.babs.valueChanged.connect(self.joint_b)
        self.cabs.valueChanged.connect(self.joint_c)
        self.dabs.valueChanged.connect(self.joint_d)
        self.eabs.valueChanged.connect(self.joint_e)
        self.xcoord.textEdited.connect(self.x_location)
        self.ycoord.textEdited.connect(self.y_location)
        self.zcoord.textEdited.connect(self.z_location)
        self.alfacoord.textEdited.connect(self.alfa_location)
        self.betacoord.textEdited.connect(self.beta_location)
        self.gammacoord.textEdited.connect(self.gamma_location)
 
        ########################
        self.show()
        self.var.animation_seq()
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

    def joint_a(self):
        self.absolute_a = str(self.aabs.value())
        self.text_aabs.setText(self.absolute_a)
        self.graph.set_active_motor(0,self.absolute_a)
        print(self.absolute_a)

    def joint_b(self):
        self.absolute_b = str(self.babs.value())
        self.text_babs.setText(self.absolute_b)
        self.graph.set_active_motor(2,self.absolute_b)
        print(self.absolute_b)

    def joint_c(self):
        self.absolute_c = str(self.cabs.value())
        self.text_cabs.setText(self.absolute_c)
        self.graph.set_active_motor(3,self.absolute_c)
        print(self.absolute_c) 

    def joint_d(self):
        self.absolute_d = str(self.dabs.value())
        self.text_dabs.setText(self.absolute_d)
        self.graph.set_active_motor(4,self.absolute_d)
        print(self.absolute_d)
    
    def joint_e(self):
        self.absolute_e = str(self.eabs.value())
        self.text_eabs.setText(self.absolute_e)
        self.graph.set_active_motor(5,self.absolute_e)
        print(self.absolute_e)  

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
