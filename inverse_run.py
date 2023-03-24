import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from Inverse_Kinematics.forward import *
from static.presets_robot_models import preset_models
import kinematics
from inverse import with_torch
class inverse_ai():
    def __init__(self, coordinates = None):
        self.alpha, self.theta, self.radius, self.dists, self.active, self.limits = get_DH_params(preset_models["HangingArm"])
        self.For_model = kinematics.Kinematics("HangingArm")
        self.model = torch.load(r"Inverse_Kinematics\kine_models\HangingArm.pt")
        if coordinates == None:
            self.gen_test()
        else:
            self.transform(coordinates)
    
    def transform(self, coord):
        n = len(coord)
        self.motor_positions = self.process(n, coord)

    def computing(self, input):
        input = np.array(input)
        input = input[:3][None,:]
        pred = self.model(torch.Tensor(input)).detach()
        return pred[0]

    def process(self, n, inputs):
        #######################
        pred = self.model(torch.Tensor(inputs[n:n+1])).detach()
        ######################
        self.For_model.set_active(pred[0])
        self.For_model.update_position()
        pos_pred = self.For_model.run_forward().round(2)
        ######################
        print("--- x y z ---")
        print(inputs[n].round(2))
        print(pos_pred)
        ######################
        print("--- torch ---")
        pred_xys = with_torch().forward_from_active(self.For_model, pred)
        print(pred_xys[0,0].numpy().round(2))
        print("--- motor positions ---")
        print(pred.numpy().round(2)[0])
        return pred


    def gen_test(self):
        inputs, target = self.For_model.generate_maps(n=1)
        #mot_pos = np.array([alpha,theta,radius,dists])
        n= np.random.randint(len(inputs))
        #######################
        pred = self.model(torch.Tensor(inputs[n:n+1])).detach()
        ######################
        self.For_model.set_active(pred[0])
        self.For_model.update_position()
        pos_pred = self.For_model.run_forward().round(2)
        #######################
        self.For_model.set_active(target[n])
        pos_targ = self.For_model.run_forward().round(2)
        ########################
        print("--- x y z ---")
        print(inputs[n].round(2))
        print(pos_targ)
        print(pos_pred )
        print("--- torch ---")
        pred_xys = with_torch().forward_from_active(self.For_model, target[n:n+1])
        print(pred_xys[0,0].numpy().round(2))
        pred_xys = with_torch().forward_from_active(self.For_model, pred)
        print(pred_xys[0,0].numpy().round(2))
        print("--- motor positions ---")
        print(target[n].round(2))
        print(pred.numpy().round(2)[0])


if __name__ == "__main__":
    inverse_ai()