from static.presets_robot_models import preset_models
from kinematics import get_DH_params
from functools import partial

class dynamic_gui:
    def __init__(self, backend):
        self.backend = backend
        self.main = self.backend.main
        self.first = 0
        self.val = partial(self.slider_change)
        for i,key in enumerate(preset_models.keys()):
            self.main.robot_options.insertItem(i,key)
        
    def visible_path(self, animate = False, click = False):
        try: self.backend.endpositions
        except: self.backend.model_build()
        if self.main.vis_path == False and self.first != 0 and click != True and animate != True:
            self.main.graph.hide_path()
        elif self.main.vis_path == True and animate == False and click == False or click == True and self.main.vis_path == False or self.first == 0:
            self.main.graph.draw_path(self.backend.endpositions, self.first)
        if click == False and animate == False or self.first == 0:
            self.main.vis_path = not self.main.vis_path
        self.first = 1

    def clicked(self):
        self.robot = self.main.robot_options.currentItem()
        self.main.graph.model_param(self.robot.text())
        self.backend.motor_list = []
        self.backend.coord_list = []
        self.main.motorlist.clear()
        self.main.coordlist.clear()
        self.slider_change()
        motorlist = self.main.kinematics.motorscan()
        self.backend.model_build()
        if self.main.vis_path == False and self.first != 0:
            self.visible_path(click = True)
        self.main.graph.update_position()
        self.main.graph.update()

    def slider_change(self):
        self.backend.copying()
        self.backend.slider_visual()
    
    ################################################################################
    ####################### absolute movement ######################################
    ################################################################################

    def slider_input(self):
        try:    reference = self.main.graph.limits.index([])
        except: reference = len(self.main.graph.limits)
        for i in range(len(self.backend.limits)):
            if i < reference:
                self.backend.absolute[i] = str(self.backend.joint[i].value())
                self.backend.slider_text[i].setText(self.backend.absolute[i])
                self.main.graph.set_active_motor(i,self.backend.absolute[i])
            else:
                self.backend.absolute[i] = str(self.backend.joint[i].value())
                self.backend.slider_text[i].setText(self.backend.absolute[i])
                self.main.graph.set_active_motor(i+1,self.backend.absolute[i])
    
    def location(self):
        for i,var in enumerate(self.backend.location_input):
            if var.text() != "":
                try: int(var.text())
                except: return
                self.backend.location_3d[i] = int(var.text())

def val(i, pos, limits, radius, theta, active):
    if i ==0 and pos != False:
        limits.pop(pos)
        active.pop(pos)
        radius.pop(pos)
        theta.pop(pos)
    if i < len(limits): 
        type = active[i]
        if type == "r":
            return str(radius[i])
        elif type == "t":
            return str(theta[i])
