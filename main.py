#libraries
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np 

#file based imports
from App_GUI import Ui_MainWindow
from serial_com import *

from json_compiler import *
from kinematics import Kinematics
try:
    from cameras import *
except:
    print("Pleora Library not Installed")


class app_stitching(QMainWindow, Ui_MainWindow):
    def __init__(self, port = None):
        super().__init__()

        ## Initialisation of GUI ##

        self.setupUi(self)
        
        ## Threaded Camera Left## 
        #self.cam_l = Feed("2BA200004267") ## Number represents the camera adress on the computer ##

        #self.cam_l.start()
        #self.cam_l.ImageUpdate.connect(self.image_update_left)
       
        ## Threaded Camera Right## 
        #self.cam_r = Feed(0) ## Number represents the camera adress on the computer ##

        #self.cam_r.start()
        #self.cam_r.ImageUpdate.connect(self.image_update_right)

        ## conection to interface to create matplot visual#
        self.graph = self.visual

        ## visual locations of the graph when initializing ##
        self.absolute_a = "0"
        self.text_aabs.setText("0")
        self.absolute_b = "100"
        self.text_babs.setText("100")
        self.absolute_c = "270"
        self.text_cabs.setText("270")
        self.absolute_d = "80"
        self.text_dabs.setText("80")
        self.absolute_e = "40"
        self.text_eabs.setText("40")
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
        self.recording.clicked.connect(self.append_motor)
        self.compiling.clicked.connect(self.json_file)
        self.remove_motor.clicked.connect(self.handle_motorlist)
        self.executing.clicked.connect(self.run_json)

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
        self.motor_list = []

        self.com = serial_bridge()

        ## json compiler initiation
        self.file = json_handler()


        ########################
        self.kinematics = Kinematics()
        
        motorlist = self.kinematics.motorscan()
        self.load_motor_table(motorlist)

        self.kinematics.orientation= True
        endpositions = self.kinematics.forward_list(motorlist)
        self.load_realpos_table(endpositions)
        
        ## A second figure pops up somewhere, fix this 
        self.show()
        ###########################
        self.graph.draw_path(endpositions)
        self.kinematics.orientation= False
        all_positions = self.kinematics.forward_list(motorlist, end_only=False)
        self.graph.animate(all_positions)
        ##########################

    def main_home(self):
        self.com.home()
        print("UI update to home position")

    
    ################################################################################
    ####################### coordinate list ########################################
    ################################################################################

    def handle_coordlist(self):
        if self.coord_list == []:
            print("no coords")
            return
        self.coord_list.pop()
        last = self.coordlist.count()
        self.coordlist.takeItem(last-1)

    def append_coord(self):
        self.coord_list.append([self.x_loc,self.y_loc,self.z_loc,self.alfa_loc,self.beta_loc,self.gamma_loc])
        self.coordlist.addItem("x: "+str(self.x_loc)+" y: "+str(self.y_loc)+" z: "+str(self.z_loc)+ " α : "+str(self.alfa_loc)+ " β: "+str(self.beta_loc)+ " γ: "+str(self.gamma_loc))

    #################################
    def handle_motorlist(self):
        if self.motor_list == []:
            print("no motor positions")
            return
        self.motor_list.pop()
        last = self.motorlist.count()
        self.motorlist.takeItem(last-1)

    def append_motor(self):
        self.motor_list.append([self.absolute_a, self.absolute_b,self.absolute_c,self.absolute_d,self.absolute_e])
        self.motorlist.addItem("A: "+str(self.absolute_a)+" B: "+str(self.absolute_b)+" C: "+str(self.absolute_c)+ " D : "+str(self.absolute_d)+ " E: "+str(self.absolute_e))
    
    def load_motor_table(self, motor_list):

        self.motor_list = motor_list
        #TODO: empty the list before you write to it
        for mot in motor_list:
            a,b,c,d,e = mot
            self.motorlist.addItem("A: "+str(a)+" B: "+str(b)+" C: "+str(c)+ 
                                  " D : "+str(d)+ " E: "+str(e))
    
    def load_realpos_table(self, endpositions):

        self.coord_list = endpositions
        ## Render end effector location and orientation
        for end in endpositions:
            x,y,z,a,b,g = np.round(end,1)
            self.coordlist.addItem(
                "x: "+str(x)+" y: "+str(y)+ " z: "+str(z) + 
               " α: "+str(a)+" β: "+str(b)+ " γ: "+str(g)
                )
    
    ########################################

    def keyPressEvent(self, e):
        print(e.key())
        #every key has a digit value press any key when screen is open to see value
        #when entered into a label enter needs to be pressed 
        delete = 16777216 #esc to undo
        enter = 16777220 #enter to execute
        if e.key() == delete:
            print("removing last coordinate")
            self.handle_coordlist()

        if e.key() == enter:
            print("starting execute of absolute coordinates.")
            self.query()

    ################################################################################

    def json_file(self):
        self.file.transfer(self.motor_list)
    
    def run_json(self):

        motor_pos = self.file.unpack()

        # Breaking Json formatted differently
        recording = self.file.unpack()
        self.com.home()
        for y,ls in enumerate(recording):
            for i in range(len(ls)):
                self.com.send_move(ls[i]["0"])
                self.com.send_move(ls[i]["1"])
                self.com.send_move(ls[i]["2"])

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
    main = app_stitching()

    sys.exit(app.exec())

 
   
    