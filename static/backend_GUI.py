import copy
import numpy as np
from functools import partial
from .robot_interface import dynamic_gui, val

class backend():
    def __init__(self, main):
        self.main = main
        self.dynamic = dynamic_gui(self)
        
        ## visual locations of the graph when initializing ##
        self.slider_text = [self.main.text_aabs, self.main.text_babs, self.main.text_cabs, self.main.text_dabs, self.main.text_eabs, self.main.text_fabs]
        self.joint=[self.main.aabs, self.main.babs, self.main.cabs, self.main.dabs, self.main.eabs, self.main.fabs]
        self.copying()
        self.slider_visual()
        self.slider_limits()


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

    def copying(self):
        self.limits = copy.deepcopy(self.main.graph.limits)
        self.active = copy.deepcopy(self.main.graph.active)
        self.radius = copy.deepcopy(self.main.graph.radius)
        self.theta = copy.deepcopy(self.main.graph.theta)
    
    def slider_limits(self):
        for i, lim in enumerate(self.limits):
            self.joint[i].setMinimum(lim[0])
            self.joint[i].setMaximum(lim[1])
            self.joint[i].setValue(int(self.absolute[i]))

    def slider_visual(self):
        self.absolute= []
        for i in range(len(self.main.graph.active)):
            try: pop_pos=self.main.graph.active.index("")
            except: pop_pos=False
            self.absolute.append(val(i, pop_pos, self.limits, self.radius, self.theta, self.active))
        for i in range(len(self.slider_text)):
            if i < len(self.absolute):
                self.slider_text[i].setText(self.absolute[i])
            else:
                self.slider_text[i].setText(None)

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
        all_positions = self.model_build()
        self.dynamic.visible_path(animate=True)
        self.main.graph.animate(all_positions)

    def model_build(self):
        motorlist = self.main.kinematics.motorscan()
        self.load_motor_table(motorlist)
        self.main.kinematics.orientation= True
        self.endpositions = self.main.kinematics.forward_list(motorlist)
        self.load_realpos_table(self.endpositions)
        self.main.kinematics.orientation= False
        all_positions = self.main.kinematics.forward_list(motorlist, end_only=False)
        return all_positions