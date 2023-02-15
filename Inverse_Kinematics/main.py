import sys
sys.path.insert(1,r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis\static/")
sys.path.insert(2,r"C:\Users\mheva\OneDrive\Documents\GitHub\Stitching_Arm_Master_Thesis/")
import numpy as np
import matplotlib.pyplot as plt
import torch
from forward import *
from presets_robot_models import preset_models
import kinematics
from inverse import with_torch

alpha, theta, radius, dists, active, limits = get_DH_params(preset_models["HangingArm"])
For_model = kinematics.Kinematics("HangingArm")
model = torch.load("../kine_models/HangingArm.pt")


inputs, target = For_model.generate_maps(n=1)
#mot_pos = np.array([alpha,theta,radius,dists])
n= np.random.randint(len(inputs))
#######################
pred = model(torch.Tensor(inputs[n:n+1])).detach()
######################
For_model.set_active(pred[0])
For_model.update_position()
pos_pred = For_model.run_forward().round(2)
#######################
For_model.set_active(target[n])
pos_targ = For_model.run_forward().round(2)
########################
print("--- x y z ---")
print(inputs[n].round(2))
print(pos_targ)
print(pos_pred )
print("--- torch ---")
pred_xys = with_torch().forward_from_active(For_model, target[n:n+1])
print(pred_xys[0,0].numpy().round(2))
pred_xys = with_torch().forward_from_active(For_model, pred)
print(pred_xys[0,0].numpy().round(2))
print("--- motor positions ---")
print(target[n].round(2))
print(pred.numpy().round(2)[0])
