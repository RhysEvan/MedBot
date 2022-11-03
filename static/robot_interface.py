from static.presets_robot_models import preset_models
from kinematics import get_DH_params

class test:
    def __init__(self, backend):
        self.backend = backend
        for i,key in enumerate(preset_models.keys()):
            self.backend.main.robot_options.insertItem(i,key)
        

    def clicked(self, model_idx):
        robot = self.backend.main.robot_options.currentItem()
        atrdal = get_DH_params(preset_models[robot.text()])
        self.backend.main.graph.alpha, self.backend.main.graph.theta, self.backend.main.graph.radius, self.backend.main.graph.dists, self.backend.main.graph.active, self.backend.main.graph.limits = atrdal
        self.backend.main.kinematics.alpha, self.backend.main.kinematics.theta, self.backend.main.kinematics.radius, self.backend.main.kinematics.dists, self.backend.main.kinematics.active, self.backend.main.kinematics.limits = atrdal
        self.backend.main.graph.update_position()
        self.backend.main.graph.update()