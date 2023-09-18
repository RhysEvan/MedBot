from Inverse_Kinematics.kinematics import get_DH_params
from functools import partial
from Inverse_Kinematics.Inverse_run import inverse_ai
from PyQt5.QtWidgets import *
import copy


class dynamic_gui:
    def __init__(self, backend):
        self.no_click = True
        self.first_param = True
        self.backend = backend
        self.main = self.backend.main
        self.param = self.robot_from_presets(self.main.robot_type)
        self.first = 0
        self.val = partial(self.slider_change)
        self.model_list()
        self.param_load()
        self.inverse = inverse_ai(self)

    def show_path(self):
        self.main.vis_path = not self.main.vis_path
        self.visible_path()
    
    def visible_path(self):
        try: self.backend.endpositions
        except: self.backend.model_build()
        if self.main.vis_path == False:
            self.main.graph.hide_path()
        elif self.main.vis_path == True:
            self.main.graph.draw_path(self.backend.endpositions, self.first)
        self.first = 1

    def clicked(self):
        self.no_click = False
        self.robot = self.main.robot_options.currentItem()
        self.main.graph.kin.model_type(self.robot.text())
        self.param_load()
        self.backend.motor_list = []
        self.backend.coord_list = []
        self.main.motorlist.clear()
        self.main.coordlist.clear()
        self.slider_change()
        self.backend.slider_limits()
        motorlist = self.main.kinematics.motorscan()
        self.backend.model_build()
        self.main.graph.kin.update_position()
        self.main.graph.update()
        if self.main.vis_path == True:
            self.main.graph.draw_path(self.backend.endpositions, self.first)

    def slider_change(self):
        self.backend.copying()
        self.backend.slider_visual()
    
    ################################################################################
    ########################## Motor movement ######################################
    ################################################################################

    def slider_motor(self):
        try:    reference = self.main.graph.kin.limits.index([])
        except: reference = len(self.main.graph.kin.limits)
        for i in range(len(self.backend.limits)):
            self.backend.absolute[i] = str(self.backend.joint[i].value())
            self.backend.slider_text[i].setText(self.backend.absolute[i])
            if i < reference: 
                self.main.graph.set_active_motor(i,self.backend.absolute[i])
            else:
                self.main.graph.set_active_motor(i+1,self.backend.absolute[i])
    
    def location(self):
        for i,var in enumerate(self.backend.location_input):
            if var.text() != "":
                try: int(var.text())
                except: return
                self.backend.location_3d[i] = int(var.text())

    #################################################################################
    ############################ End movement #######################################
    #################################################################################

    def slider_end(self):
        mem = []
        try:    reference = self.main.graph.kin.limits.index([])
        except: reference = len(self.main.graph.kin.limits)
        for i in range(len(self.backend.cartesian)):
            mem.append(self.backend.cartesian[i].value())
        motor_positions = self.inverse.computing(mem)
        for i in range(len(self.backend.limits)):
            if i < reference: 
                self.main.graph.set_active_motor(i,motor_positions[i])
            else:
                self.main.graph.set_active_motor(i+1,motor_positions[i])


    #################################################################################
    ########################### DH parameters #######################################
    #################################################################################

    def change_alpha(self,alpha):
        try: self.param["alpha"] = string_to_list(alpha)
        except: return
        
    def change_theta(self, theta):
        try: self.param["theta"] = string_to_list(theta)
        except: return
    
    def change_radius(self, radius):
        try: self.param["radius"] = string_to_list(radius)
        except: return
    
    def change_dists(self, dists):
        try: self.param["dists"] = string_to_list(dists)
        except: return

    def change_active(self, active):
        try: self.param["active"] = string_to_list_active(active)
        except: return
    
    def change_limits(self,limits):
        print(self.param)
        try: self.param["limits"] = string_to_list2(limits)
        except: print(self.param["limits"]) ; return
            
    def param_load(self):
        if not self.first_param:
            self.param = self.robot_from_presets(self.robot.text())
        self.main.DH_param_1.setText(str(self.param["alpha"]))
        self.main.DH_param_2.setText(str(self.param["theta"]))
        self.main.DH_param_3.setText(str(self.param["radius"]))
        self.main.DH_param_4.setText(str(self.param["dists"]))
        self.main.DH_param_5.setText(str(self.param["active"]))
        self.main.DH_param_6.setText(str(self.param["limits"]))
        self.first_param = False

    def json_type(self):
        if self.no_click:
            name , done = QInputDialog.getText(
                    self.main, 'Saving with name', 'Enter the name:', QLineEdit.Normal, self.main.robot_type)
        else:
            name , done = QInputDialog.getText(
            self.main, 'Saving with name', 'Enter the name:', QLineEdit.Normal, self.robot.text())
        if done:
            update = self.dictonary_handler(name)
            self.main.file.transfer(update)
            self.model_list()
    
    def robot_from_presets(self,name):
        robots = self.presets()
        robot = robots[name]
        return robot

    def presets(self):
        self.main.file.filename = "./static/presets.json"
        return self.main.file.unpack()

    def dictonary_handler(self, name):
        robot = self.main.file.unpack()
        robot[name] = self.param
        return robot
    
    def model_list(self):
        robots = self.presets()
        self.main.robot_options.clear()
        for i,key in enumerate(robots.keys()):
            self.main.robot_options.insertItem(i,key)
    
    def update_visual(self):
        param = self.main.kinematics.DH_params(self.param)
        print(param)
        self.main.graph.kin.model_param(param)
        self.backend.motor_list = []
        self.backend.coord_list = []
        self.main.motorlist.clear()
        self.main.coordlist.clear()
        self.slider_change()
        self.backend.slider_limits()
        motorlist = self.main.kinematics.motorscan()
        self.backend.model_build()
        self.main.graph.kin.update_position()
        self.main.graph.update()
        if self.main.vis_path == True:
            self.main.graph.draw_path(self.backend.endpositions, self.first)

def val(i, pos, limits, radius, theta, active):
    if i ==0 and pos != False:
        limits.pop(pos)  ;  active.pop(pos)  ;  radius.pop(pos)  ;  theta.pop(pos)
    if i < len(limits): 
        type = active[i]
        if type == "r":
            return str(radius[i])
        elif type == "t":
            return str(theta[i])

def string_to_list(x):
    return list(map(int, x[1:-1].replace(" ","").split(',')))

def string_to_list2(x):
    y = []
    res = list(x[2:-2].replace(" ","").split('],['))
    print(res)
    y = []
    for r in res:
        try: y.append(list(map(int,r.split(","))))
        except: 
            if r.strip() == '':
                y.append([])
            else:
                continue
    return y

def string_to_list_active(x):
    return list(map(str.strip, x.strip('][').replace("'", '').split(',')))