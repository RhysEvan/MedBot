import sys
import time

import serial
import serial.tools.list_ports as port_list
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from App_GUI import Ui_MainWindow

def connect_to_ports(find_name):
    all_ports = list(port_list.comports())
    pos_ports = [p.device for p in all_ports  if "Arduino" in p.description]
    [print(p.description) for p in all_ports]

    if pos_ports == []:
        all_ports = list(port_list.comports())
        ard = all_ports[0].device
        return ard

    ## Search for Suitable Port
    print(pos_ports)
    for port in pos_ports: 
        print(".")
        try:      ard = serial.Serial(port, 115200, timeout=0.1)
        except:   continue
        print("trying", port, "...", end="")
        response = read_info(ard)
        print(response, "...", end="")
        if response == find_name: 
            print("Port Found: ", port)
            break
        else:  
            ard.close()
            ard = None
    print("")

    return ard

def read_info(ard):

    for _ in range(10): 
        response = ard.readline().decode("utf-8").split("\r")[0]
        if response == "":
            print(".",end="")
        if response == "Startup":
            print("Starting up device")
            time.sleep(.1)
            break         
    ard.write(b"Info\r")
    Info = ard.readline().decode("utf-8").split("\r")[0]
    print("Device Info: "+ Info)  
    return Info

class app_stitching(QMainWindow, Ui_MainWindow):
    def __init__(self, port = None):
        super(app_stitching, self).__init__()

        self.setupUi(self)

        self.absolute_a = "0"
        self.absolute_b = "0"
        self.absolute_c = "0"
        self.absolute_d = "0"
        self.absolute_e = "0"

        self.x_loc = "0"
        self.y_loc = "0"
        self.z_loc = "0"

        self.Homing.clicked.connect(self.home)
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

        if port is None:
            port = connect_to_ports("")
            self.device = serial.Serial(port, 155200, timeout=0.1)
            print(self.device)
        else:
            self.device = serial.Serial(port, 155200, timeout=0.1)
        
        #self.home()

    @pyqtSlot()
    def home(self):
        print("homing")
        self.device.write(bytearray("$X\r\n","utf-8"))
        self.device.write(bytearray("$F 100\r\n","utf-8"))
        self.device.write(bytearray("$HX\r\n","utf-8"))
        #self.device.write(bytearray("$HY\r\n","utf-8"))
    
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
        if e.key() == Qt.Key_delete:
            print("removing last coordinate")
            self.handle_list()

        if e.key() == Qt.Key_Enter:
            print("starting execute of absolute coordinates.")
            self.query()

    ################################################################################
    ########################## movement query ######################################
    ################################################################################

    def query(self):
        print("query starting")
        self.device.write(bytearray("$G1 x "+self.absolute_a+" y "+self.absolute_b+"\r","utf-8"))
        self.device.write(bytearray("$G1 z "+self.absolute_c+" a "+self.absolute_d+"\r","utf-8"))
        self.device.write(bytearray("$G1 b "+self.absolute_e))

    ################################################################################
    ####################### absolute movement ######################################
    ################################################################################

    @pyqtSlot()
    def joint_a(self, a):
        self.absolute_a = a

    @pyqtSlot()
    def joint_b(self, b):
        self.absolute_b = b

    @pyqtSlot()
    def joint_c(self, c):
        self.absolute_c = c

    @pyqtSlot()
    def joint_d(self, d):
        self.absolute_d = d
    
    @pyqtSlot()
    def joint_e(self, e):
        self.absolute_e = e  
    
    @pyqtSlot()
    def x_location(self, x):
        self.x_loc = x

    @pyqtSlot()
    def y_location(self, y):
        self.y_loc = y

    @pyqtSlot()
    def z_location(self, z):
        self.z_loc = z

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = app_stitching()
    main.show()
    sys.exit(app.exec())