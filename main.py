#libraries
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

#file based imports
from app_GUI import Ui_MainWindow
from serial_com import *
from cameras import *


class app_stitching(QMainWindow, Ui_MainWindow):
    def __init__(self, port = None):
        super().__init__()

        ## Initialisation of GUI ##

        self.setupUi(self)

        
        ## Threaded Camera Left## 
        self.cam_l = Feed_Left(0) ## Number represents the camera adress on the computer ##

        self.cam_l.start()
        self.cam_l.ImageUpdateLeft.connect(self.image_update_left)
       
        ## Threaded Camera Right## 
        self.cam_r = Feed_Right(1) ## Number represents the camera adress on the computer ##

        self.cam_r.start()
        self.cam_r.ImageUpdateRight.connect(self.image_update_right)

        ## conection to interface to create matplot visual#
        self.graph = self.visual
        
        ## visual locations of the graph when initializing ##
        self.absolute_a = "0"
        self.absolute_b = "100"
        self.absolute_c = "270"
        self.absolute_d = "80"
        self.absolute_e = "40"

        ## initial values for the recording list of xyz values ##
        self.x_loc = "0"
        self.y_loc = "0"
        self.z_loc = "0"
        self.alfa_loc = "0"
        self.beta_loc = "0"
        self.gamma_loc = "0"

        ## call to button functions and their forward to internal functions ##
        self.Homing.clicked.connect(self.main_home)
        self.Submit.clicked.connect(self.append_coord)
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

        self.coord_list = []

        self.com = serial_bridge()

    @pyqtSlot()
    def main_home(self):
        self.com.home()
        print("UI update to home position")

    
    ################################################################################
    ####################### coordinate list ########################################
    ################################################################################

    @pyqtSlot()
    def append_coord(self):
        print("coord-list")
        self.coord_list.append([self.x_loc,self.y_loc,self.z_loc,self.alfa_loc,self.beta_loc,self.gamma_loc])
        self.coordlist.addItem("x: "+str(self.x_loc)+" y: "+str(self.y_loc)+" z: "+str(self.z_loc)+ " α : "+str(self.alfa_loc)+ " β: "+str(self.beta_loc)+ " γ: "+str(self.gamma_loc))
        print(self.coord_list)

    def handle_list(self):
        self.coord_list.pop()
        last = self.coordlist.count()
        self.coordlist.takeItem(last-1)
        print(self.coord_list)

    def keyPressEvent(self, e):
        print(e.key())
        #every key has a digit value press any key when screen is open to see value
        #when entered into a label enter needs to be pressed 
        delete = 16777216 #esc to undo
        enter = 16777220 #enter to execute
        if e.key() == delete:
            print("removing last coordinate")
            self.handle_list()

        if e.key() == enter:
            print("starting execute of absolute coordinates.")
            self.query()

    ################################################################################
    ########################## movement query ######################################
    ################################################################################

    def query(self):
        print("query starting")
        print("arduino commands currently turned off, GRBL settings not stable yet. 21/7")
        self.com.send_move("x "+self.absolute_a+" y "+self.absolute_b)
        self.com.send_move("z "+self.absolute_c+" a "+self.absolute_d)
        self.com.send_move("b "+self.absolute_e)
        
    ################################################################################
    ####################### absolute movement ######################################
    ################################################################################

    def joint_a(self):
        self.absolute_a = self.aabs.value()
        self.graph.set_motor(0,self.absolute_a)
        print(self.absolute_a)

    def joint_b(self):
        self.absolute_b = self.babs.value()
        self.graph.set_motor(2,self.absolute_b)
        print(self.absolute_b)

    def joint_c(self):
        self.absolute_c = self.cabs.value()
        self.graph.set_motor(3,self.absolute_c)
        print(self.absolute_c) 

    def joint_d(self):
        self.absolute_d = self.dabs.value()
        self.graph.set_motor(4,self.absolute_d)
        print(self.absolute_d)
    
    def joint_e(self):
        self.absolute_e = self.eabs.value()
        self.graph.set_motor(5,self.absolute_e)
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
    main = app_stitching()
    main.show()
    sys.exit(app.exec())