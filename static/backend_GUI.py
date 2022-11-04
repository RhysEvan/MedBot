import copy
import numpy as np
from functools import partial
from .robot_interface import dynamic_gui

class backend():
    def __init__(self, main):
        self.main = main
        self.dynamic = dynamic_gui(self)
        ## visual locations of the graph when initializing ##
        self.absolute_a = "0" # change for self.val(1) for example. You want to make dynamic code that can
        self.main.text_aabs.setText("0") #open the models and change these values with the change of model
        self.absolute_b = "100" # depending on the active value being r or t you need to take a different string value
        self.main.text_babs.setText("100") # buffer for "" need to be added as well, plus a new variable for f too.
        self.absolute_c = "270"
        self.main.text_cabs.setText("270")
        self.absolute_d = "80"
        self.main.text_dabs.setText("80")
        self.absolute_e = "40"
        self.main.text_eabs.setText("40")
        self.absolute_f = "0"
        self.main.text_fabs.setText("0")
        ## initial values for the recording list of xyz values ##
        self.x_loc = "0"
        self.y_loc = "0"
        self.z_loc = "0"
        self.alfa_loc = "0"
        self.beta_loc = "0"
        self.gamma_loc = "0"
        #
        self.coord_list = []
        self.motor_list = []


    ################################################################################
    ########################## movement query ######################################
    ################################################################################

    def query(self):
        print("query starting")
        print("arduino commands currently turned off, GRBL settings not stable yet. 21/7")
        print(self.absolute_a)
        self.main.com.send_move("x "+self.absolute_a+" y "+self.absolute_b)
        self.main.com.send_move("z "+self.absolute_c+" a "+self.absolute_d)
        self.main.com.send_move("b "+self.absolute_e)
    
    ################################################################################
    ####################### coordinate list ########################################
    ################################################################################
    
    def handle_coordlist(self):
        if self.coord_list == []:
            print("no coords")
            return
        self.coord_list.pop()
        last = self.main.coordlist.count()
        self.main.coordlist.takeItem(last-1)

    def append_coord(self):
        self.coord_list.append([self.x_loc,self.y_loc,self.z_loc,self.alfa_loc,self.beta_loc,self.gamma_loc])
        self.main.coordlist.addItem("x: "+str(self.x_loc)+" y: "+str(self.y_loc)+" z: "+str(self.z_loc)+ " α : "+str(self.alfa_loc)+ " β: "+str(self.beta_loc)+ " γ: "+str(self.gamma_loc))

    #################################
    def handle_motorlist(self):
        if self.motor_list == []:
            print("no motor positions")
            return
        self.motor_list.pop()
        last = self.main.motorlist.count()
        self.main.motorlist.takeItem(last-1)

    def append_motor(self):
        self.motor_list.append([self.absolute_a, self.absolute_b,self.absolute_c,self.absolute_d,self.absolute_e])
        self.main.motorlist.addItem("A: "+str(self.absolute_a)+" B: "+str(self.absolute_b)+" C: "+str(self.absolute_c)+ " D : "+str(self.absolute_d)+ " E: "+str(self.absolute_e))
    
    def load_motor_table(self, motor_list):

        self.motor_list = motor_list
        #TODO: empty the list before you write to it
        for mot in motor_list:
            s = ""
            for i,val in enumerate(mot):
                part = self.motorliststring(i,val)
                s += part
            self.main.motorlist.addItem(s)
    
    def motorliststring(self, i, val):
        mot_id = ["A: ", "B: ", "C: ", "D: ", "E: "]
        st = mot_id[i]+str(val)+" "
        return st

    def load_realpos_table(self, endpositions):

        self.coord_list = endpositions
        ## Render end effector location and orientation
        for end in endpositions:
            x,y,z,a,b,g = np.round(end,1)
            self.main.coordlist.addItem(
                "x: "+str(x)+" y: "+str(y)+ " z: "+str(z) + 
               " α: "+str(a)+" β: "+str(b)+ " γ: "+str(g)
                )

    ################################################################################

    def json_file(self):
        self.main.file.transfer(self.motor_list)
    
    def run_json(self):

        motor_pos = self.main.file.unpack()

        # Breaking Json formatted differently
        recording = self.main.file.unpack()
        self.main.com.home()
        for y,ls in enumerate(recording):
            for i in range(len(ls)):
                self.main.com.send_move(ls[i]["0"])
                self.main.com.send_move(ls[i]["1"])
                self.main.com.send_move(ls[i]["2"])
        
    def main_home(self):
        self.main.com.home()
        print("UI update to home position")

    def initiate_animate(self):
        print('updating and animating')
        self.animation_seq()
    
    def animation_seq(self):
        
        motorlist = self.main.kinematics.motorscan()
        self.load_motor_table(motorlist)

        self.main.kinematics.orientation= True
        self.endpositions = self.main.kinematics.forward_list(motorlist)
        self.load_realpos_table(self.endpositions)
        self.main.kinematics.orientation= False
        self.dynamic.visible_path(animate=True)
        all_positions = self.main.kinematics.forward_list(motorlist, end_only=False)
        self.main.graph.animate(all_positions)
