
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from functools import partial
import torch
from scipy.spatial.transform import Rotation as R
import copy

import static.presets_robot_models as presets_robot_models


class Kinematics():
    
    def __init__(self, robot):

        self.orientation = False
        self.model_param(robot)
        self.update_position()    

    def model_param(self,robot):
        atrdal = get_DH_params(presets_robot_models.preset_models[robot])
        parameters = copy.deepcopy(atrdal)
        self.alpha, self.theta, self.radius, self.dists, self.active, self.limits = parameters

    def update_position(self, DH = None):
        
        if DH is None:   DH = self.get_motor_positions()
        self.position = forward_all(*DH)         

        return self.position
    
    def run_forward(self, DH=None, end_only=True):

        
        if DH is None:   DH = self.get_motor_positions()

        position = forward_all(*DH) 

        if end_only:
            position = position[:3,-1]    

        if self.orientation:
            DH = self.get_motor_positions()
            orient = self.forward_orientation(DH)
            position = np.append(position,orient)
        
        return position

    def forward_orientation(self,DH):

        Tfs = make_transforms(*DH)[-1]
        orient = R.from_matrix(Tfs[:3,:3]).as_euler("XYZ")

        if orient[0]<0 or orient[2]<0:  
            orient[[0,2]] = orient[[0,2]] + np.pi
            orient[[1]]   = np.pi - orient[[1]] 

        orient = np.degrees(orient)

        return orient

    def forward_list(self, active_motor_pos_list, end_only=True):

        position_list = []
        for active in active_motor_pos_list:

            self.set_active(active)
            position = self.run_forward(end_only=end_only)
            position_list.append(position)

        return position_list
    
    #########
    def random_positions(self,):
        
        mot_pos = self.get_motor_positions()
        active = self.active
        limits = self.limits
        
        for n,limit in enumerate(limits):
            if active[n]=="": continue
            if len(limit)==0: continue

            v = np.random.uniform(limit[0],limit[1])

            if active[n]=="t":            mot_pos[1,n] = v
            if active[n]=="r":            mot_pos[2,n] = v

        return mot_pos
    
    def random_positions_paths(self,pos):
        mot_pos = self.get_motor_positions()
        active = self.active
        limits = self.limits
        skip = False
        for n,limit in enumerate(limits):
            i = 0
            if active[n]=="": 
                skip = True
                continue
            if len(limit)==0: continue
            while i == 0:
                val = (limit[1]+limit[0])/100
                v = np.random.uniform(-val,val)
                if not skip:
                    if pos[n]+v > limit[0] and pos[n]+v < limit[1]:
                        pos[n] = pos[n]+v
                        if active[n]=="t":            mot_pos[1,n] = v
                        if active[n]=="r":            mot_pos[2,n] = v
                        i = 1
                else:
                    if pos[n-1]+v > limit[0] and pos[n-1]+v < limit[1]:
                        pos[n-1] = pos[n-1]+v
                        if active[n]=="t":            mot_pos[1,n] = v
                        if active[n]=="r":            mot_pos[2,n] = v
                        i = 1
        return mot_pos, pos
    
    def generate_maps(self,n=100):
        
        inputs, target = [],[]
        for x in range(n):

            mot_pos = self.random_positions()
            self.set_motor_positions(mot_pos)
            act_pos = self.get_active()
            end_pos = self.run_forward()

            inputs.append(end_pos)
            target.append(act_pos)

        inputs = np.array(inputs)
        target = np.array(target)    

        return inputs, target

    def generate_paths(self,n=100):
        prev_pos = self.random_positions()
        self.set_motor_positions(prev_pos)
        prev_pos = self.get_active()
        inputs, target = [],[]
        for x in range(n):
            
            mot_pos, prev_pos= self.random_positions_paths(prev_pos)
            self.set_motor_positions(mot_pos)
            act_pos = self.get_active()
            end_pos = self.run_forward()
            inputs.append(end_pos)
            target.append(act_pos)

        inputs = np.array(inputs)
        target = np.array(target)    

        return inputs, target

    ########
    def motorscan(self):

        limits = self.limits

        mot_pos = self.centermotor()
        self.set_motor_positions(mot_pos)
        act_pos = self.get_active()

        motor_list = [act_pos]
        i = 0
        for n,limit in enumerate(limits):
            if len(limit)==0: continue

            #########################
            l0,l1 = limit
            act_pos[i] = l0
            motor_list.append(act_pos.copy())
            #########################
            for ap in np.linspace(l0,l1,10):
                act_pos[i] = ap
                motor_list.append(act_pos.copy())

            act_pos[i] = l1
            motor_list.append(act_pos.copy())
            #########################
            mid = (l1 + l0)/2
            for ap in np.linspace(l1,mid,5):
                act_pos[i] = ap
                motor_list.append(act_pos.copy())
            act_pos[i] = mid
            motor_list.append(act_pos.copy())
            ##########################
            i = i+1

        return motor_list

    def centermotor(self):

        mot_pos = self.get_motor_positions()
        active = self.active
        limits = self.limits
        
        for n,limit in enumerate(limits):
            if active[n]=="": continue
            if len(limit)==0: continue

            v = limit[0] + limit[1] / 2

            if active[n]=="t":            mot_pos[1,n] = v
            if active[n]=="r":            mot_pos[2,n] = v

        return mot_pos
    
    ########
    def get_motor_positions(self):
        
        alpha = self.alpha
        theta = self.theta
        radius = self.radius
        dists = self.dists
        active = self.active
        limits = self.limits

        mot_pos = np.array([alpha,theta,radius,dists])
        return mot_pos
        
    def get_active(self):
        
        mot_pos = self.get_motor_positions()
        active = self.active

        out_pos = []
        for n,a in enumerate(active):
            if a=="": continue
            if a=="t":          v = mot_pos[1,n]    
            if a=="r":          v = mot_pos[2,n]
            #else: continue
            out_pos.append(v)

        return np.array(out_pos)

    ########
    def set_active(self, new_active):
        
        mot_pos = self.get_motor_positions()
        active = self.active
        i = 0
        for n in range(mot_pos.shape[1]):
            if active[n]=="": continue
            v = new_active[i]

            if active[n]=="r":            mot_pos[2,n] = v
            elif active[n]=="t":          mot_pos[1,n] = v
            else: continue
            i += 1
            
        self.set_motor_positions(mot_pos) 

        return mot_pos

    def set_motor_positions(self, mot_pos):
        
        a,t,r,d = mot_pos
            
        self.alpha = a
        self.theta = t
        self.radius = r 
        self.dists = d
            
        ######################################
    #  will remove references to kinematics 
    def set_active(self, new_active):
        
        mot_pos = self.get_motor_positions()
        active = self.active

        i = 0
        for n in range(mot_pos.shape[1]):
            if active[n]=="": continue
            v = new_active[i]

            if active[n]=="r":            mot_pos[2,n] = v
            elif active[n]=="t":          mot_pos[1,n] = v
            else: continue
            i += 1
            
        self.set_motor_positions(mot_pos) 

        return mot_pos
    
    def set_active_motor(self, idx, value):
        t,r = self.theta, self.radius

        if self.active[idx]=="r":            r[idx] = int(value)
        elif self.active[idx]=="t":          t[idx] = int(value)
        
        self.theta,self.radius  = t,r

 
# Base functions, should merge into object and remove references elsewhere

##############################################
def get_DH_params(model):
    
    alpha = model["alpha"]
    theta =  model["theta"]
    radius = model["radius"]
    dists = model["dists"]
    active = model["active"]
    limits = model["limits"]
    
    return alpha, theta, radius, dists, active, limits

##############################################
def forward_all(alpha, theta, radius, dists):

    DHts = make_transforms(alpha, theta, radius, dists)

    A = np.array((0,0,0,1))[None,:]
    positions = []
    for DH in DHts:
        B = np.matmul(DH,A.T).T   
        positions.append(B)

    positions = np.array(positions)
            
    return positions[:,0,:3].T

def make_transforms(alpha, theta, radius, dists):
    
    DHts = []
    T = np.eye(4)
    for i in range(len(alpha)):
        a = np.radians(alpha[i])
        t = np.radians(theta[i])
        r = radius[i]
        d = dists[i]
        DHt = DH_transform(a,t,r,d) 
        DHt = np.matmul(T,DHt)
        T = DHt.copy()
        DHts.append(DHt)

    return np.array(DHts)

def DH_transform(a,t,r,z):
    Zt = Z_transform(a,t,r,z)
    Xt = X_transform(a,t,r,z)
    DHt = np.matmul(Zt,Xt)
    return DHt

def X_transform(a,t,r,d):
    
    Xt = [[1,0,0,r],
          [0,np.cos(a), -1*np.sin(a),0],
          [0,np.sin(a),  np.cos(a),0],
          [0,0,0,1]]
    Xt = np.array(Xt)

    return Xt
    
def Z_transform(a,t,r,d):
    
    Zt = [[np.cos(t), -1*np.sin(t),0,0],
          [np.sin(t), np.cos(t), 0,0],
          [0,0,1,d],
          [0,0,0,1]]
    Zt = np.array(Zt)
    return Zt
