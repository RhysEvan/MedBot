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
        
    def visible_path(self, click = False, animate = False):
        if self.backend.main.vis_path == False and click != True and animate != True and self.first != 0:
            self.backend.main.graph.hide_path()
        elif self.backend.main.vis_path == True or click == True:
            self.backend.main.graph.draw_path(self.backend.endpositions, self.first)
            self.first = 1
        if click == False and animate == False:
            print("entered")
            self.backend.main.vis_path = not self.backend.main.vis_path
        click = False
        animate = False

    def clicked(self):
        robot = self.backend.main.robot_options.currentItem()
        atrdal = get_DH_params(preset_models[robot.text()])
        self.backend.main.graph.alpha, self.backend.main.graph.theta, self.backend.main.graph.radius, self.backend.main.graph.dists, self.backend.main.graph.active, self.backend.main.graph.limits = atrdal
        
        motorlist = self.backend.main.kinematics.motorscan()
        self.backend.load_motor_table(motorlist)
        self.backend.main.kinematics.orientation= True
        self.backend.endpositions = self.backend.main.kinematics.forward_list(motorlist)
        self.backend.load_realpos_table(self.backend.endpositions)
        self.backend.main.kinematics.orientation= False
        self.visible_path(click = True)
        
        self.backend.main.graph.update_position()
        self.backend.main.graph.update()
        self.slider_change()

    def slider_change(self):
        print("hehe")
        limits = self.backend.main.graph.limits
