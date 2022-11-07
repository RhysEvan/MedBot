import copy
import numpy as np
from functools import partial
from .robot_interface import dynamic_gui, val

class backend():
    def __init__(self, main):
        self.main = main
        self.dynamic = dynamic_gui(self)
        ## initial values for the recording list of xyz values ##
        self.x_loc = None
        self.y_loc = None
        self.z_loc = None
        self.alpha_loc = None
        self.beta_loc = None
        self.gamma_loc = None
        #
        self.coord_list = []
        self.motor_list = []
        ## visual locations of the graph when initializing ##
        self.slider_text = [self.main.text_aabs, self.main.text_babs, self.main.text_cabs, self.main.text_dabs, self.main.text_eabs, self.main.text_fabs]
        self.joint = [self.main.aabs, self.main.babs, self.main.cabs, self.main.dabs, self.main.eabs, self.main.fabs]
        self.location_3d = [self.x_loc, self.y_loc, self.z_loc, self.alpha_loc, self.beta_loc, self.gamma_loc]
        self.location_input = [self.main.xcoord, self.main.ycoord, self.main.zcoord, self.main.alphacoord, self.main.betacoord, self.main.gammacoord]
        self.copying()
        self.slider_visual()
        self.slider_limits()
        if self.main.animate:
            self.animation_seq()
        if self.main.vis_path:
            self.dynamic.visible_path()

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
            if self.absolute[i] == None:
                print("needs to become black")

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
        motor_pos = copy.deepcopy(self.absolute)
        idx = ["x ","y ","z ","a ","b ","c "]
        try: motor_pos.remove(None)
        except: motor_pos = motor_pos
        for i in range(len(motor_pos)):
            if i%2 == 0:
                try: self.main.com.send_move(idx[i]+self.absolute[i]+" "+idx[i+1]+self.absolute[i+1])
                except: self.main.com.send_move(idx[i]+self.absolute[i])
                print(self.absolute[i])
    ################################################################################
    ####################### coordinate list ########################################
    ################################################################################
    
    def handle_lists(self):
        if self.coord_list == []:
            print("no coords")
            return
        self.coord_list.pop()
        last = self.main.coordlist.count()
        self.main.coordlist.takeItem(last-1)
        if self.motor_list == []:
            print("no motor positions")
            return
        self.motor_list.pop()
        last = self.main.motorlist.count()
        self.main.motorlist.takeItem(last-1)

    def append_coord(self):
        print("appending coord with right angles")
        self.coord_string(np.array(self.location_3d))

    def calculated_coord(self):
        active = []
        for i,val in enumerate(self.joint):
            if val != None:    
                active.append(val.value())
        coordlist = np.array(active)
        self.main.kinematics.orientation= True
        new_string = self.main.kinematics.forward_list([coordlist])
        self.main.kinematics.orientation= False
        self.coord_string(new_string[0])

    def coord_string(self,end):
        try: x,y,z,a,b,g = np.round(end,1)
        except: x,y,z,a,b,g = end
        self.main.coordlist.addItem(
                "x: "+str(x)+" y: "+str(y)+ " z: "+str(z) + 
               " α: "+str(a)+" β: "+str(b)+ " γ: "+str(g)
                )

    #################################        

    def append_motor(self):
        self.motor_list.append(self.joint)
        self.calculated_coord()
        item = partial(motor_record,self.limits,self.joint)
        self.main.motorlist.addItem(item())
    
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
            self.coord_string(end)

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

    def animation_seq(self):
        all_positions = self.model_build()
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

def motor_record(limit,motor):
    item = ""
    idx = ["A","B","C","D","E","F"]
    for i in range(len(limit)):
        item += idx[i]+": "+str(motor[i].value())
    return item