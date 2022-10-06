#libraries
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

#file based code
from app_GUI import Ui_MainWindow
from Serial import *

class app_stitching(QMainWindow, Ui_MainWindow):
    def __init__(self, port = None):
        super(app_stitching, self).__init__()

        self.setupUi(self)
        self.graph = self.visual
        self.absolute_a = "0"
        self.absolute_b = "100"
        self.absolute_c = "-90"
        self.absolute_d = "80"
        self.absolute_e = "40"

        self.x_loc = "0"
        self.y_loc = "0"
        self.z_loc = "0"

        self.Homing.clicked.connect(self.main_home)
        self.Submit.clicked.connect(self.append_coord)
        self.aabs.textEdited.connect(self.joint_a)
        self.babs.textEdited.connect(self.joint_b)
        self.cabs.textEdited.connect(self.joint_c)
        self.dabs.textEdited.connect(self.joint_d)
        self.eabs.textEdited.connect(self.joint_e)
        self.xcoord.textEdited.connect(self.x_location)
        self.ycoord.textEdited.connect(self.y_location)
        self.zcoord.textEdited.connect(self.z_location)

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
        self.coord_list.append([self.x_loc,self.y_loc,self.z_loc])
        self.coordlist.addItem([self.x_loc,self.y_loc,self.z_loc])

    def handle_list(self):
        last = self.coordlist.count()
        self.coordlist.takeItem(last)

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
        positions = [self.absolute_a, self.absolute_b, self.absolute_c, self.absolute_d, self.absolute_e]
        for n,pos in enumerate(positions):
            if n > 0:
                n+=1
                self.graph.set_motor(n,pos)
            else:
                self.graph.set_motor(n,pos)

        print("arduino commands currently turned off, GRBL settings not stable yet. 21/7")
        self.com.send_com("$G1 x "+self.absolute_a+" y "+self.absolute_b)
        self.com.send_com("$G1 z "+self.absolute_c+" a "+self.absolute_d)
        self.com.send_com("$G1 b "+self.absolute_e)
        
    ################################################################################
    ####################### absolute movement ######################################
    ################################################################################

    @pyqtSlot()
    def joint_a(self):
        self.absolute_a = self.aabs.text()
        print(self.absolute_a)

    @pyqtSlot()
    def joint_b(self):
        self.absolute_b = self.babs.text()
        print(self.absolute_b)

    @pyqtSlot()
    def joint_c(self):
        self.absolute_c = self.cabs.text()
        print(self.absolute_c) 

    @pyqtSlot()
    def joint_d(self):
        self.absolute_d = self.dabs.text()
        print(self.absolute_d)
    
    @pyqtSlot()
    def joint_e(self):
        self.absolute_e = self.eabs.text()
        print(self.absolute_e)  
    
    @pyqtSlot()
    def x_location(self):
        self.x_loc = self.xcoord.text()
        print(self.x_loc)

    @pyqtSlot()
    def y_location(self):
        self.y_loc = self.ycoord.text()
        print(self.y_loc)

    @pyqtSlot()
    def z_location(self):
        self.z_loc = self.zcoord.text()
        print(self.z_loc)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = app_stitching()
    main.show()
    sys.exit(app.exec())