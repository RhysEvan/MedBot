from static.presets_robot_models import preset_models
from kinematics import get_DH_params
from functools import partial

class dynamic_gui:
    def __init__(self, backend):
        self.backend = backend
        self.first = 0
        self.val = partial(self.slider_change)
        for i,key in enumerate(preset_models.keys()):
            self.backend.main.robot_options.insertItem(i,key)
        
    def visible_path(self, animate = False, click = False):
        if self.backend.main.vis_path == False and self.first != 0 and click != True:
            self.backend.main.graph.hide_path()
        elif self.backend.main.vis_path == True or [click == True and self.backend.main.vis_path == False]:
            self.backend.main.graph.draw_path(self.backend.endpositions, self.first)
        if [animate == False or self.first == 0] and click != True:
            self.backend.main.vis_path = not self.backend.main.vis_path
        self.first = 1
        animate = False
        click = False

    def clicked(self):
        robot = self.backend.main.robot_options.currentItem()
        self.backend.main.graph.model_param(robot.text())
        self.slider_change()
        motorlist = self.backend.main.kinematics.motorscan()
        self.backend.model_build()
        self.visible_path(click = True)
        self.backend.main.graph.update_position()
        self.backend.main.graph.update()


    def slider_change(self):
        self.backend.copying()
        self.backend.slider_visual()

    def slider_input(self):
        try:    reference = self.backend.main.graph.limits.index([])
        except: reference = len(self.backend.main.graph.limits)
        for i in range(len(self.backend.limits)):
            if i < reference:
                self.backend.absolute[i] = str(self.backend.joint[i].value())
                self.backend.slider_text[i].setText(self.backend.absolute[i])
                self.backend.main.graph.set_active_motor(i,self.backend.absolute[i])
            else:
                self.backend.absolute[i] = str(self.backend.joint[i].value())
                self.backend.slider_text[i].setText(self.backend.absolute[i])
                self.backend.main.graph.set_active_motor(i+1,self.backend.absolute[i])


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
