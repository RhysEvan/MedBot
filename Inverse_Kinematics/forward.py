
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch

#############################################        
######################################

def random_positions( alpha, theta, radius, dists, active, limits):
    
    mot_pos = np.array([alpha,theta,radius,dists])
    
    for n,limit in enumerate(limits):
        if active[n]=="": continue
        if len(limit)==0: continue

        v = np.random.uniform(limit[0],limit[1])

        if active[n]=="t":            mot_pos[1,n] = v
        if active[n]=="r":            mot_pos[2,n] = v
    
    return mot_pos
             
def get_active(mot_pos, active):
    
    out_pos = []
    for n,a in enumerate(active):
        if a=="": continue
        if a=="t":          v = mot_pos[1,n]    
        if a=="r":          v = mot_pos[2,n]
        #else: continue
            
        out_pos.append(v)
    
    return np.array(out_pos)



#######################
#######################

from torch import nn
import torch.nn.functional as F

class NeuralNetworkStack(nn.Module):

    def __init__(self, n_in, n_out, nlayers=8, depth=20):
        super().__init__()
                
        Mod_list = []
        for i in range(nlayers):
            block = linear_layer(depth)
            Mod_list.append( block )
        
        self.blocks = nn.ModuleList(Mod_list)

        self.layer_first = nn.Linear(n_in, depth)
        self.layer_last = nn.Linear(depth*(nlayers+1)+n_in, n_out)
        
    def forward(self, x_in):
                
        x_next = self.layer_first(x_in)
    
        out_list = [x_in,x_next]
        for n, block in enumerate(self.blocks):
            x_next = block(x_next) 
            out_list.append(x_next)  
            
        x = torch.cat(out_list, dim=1)
        
        if self.training:
            x = F.leaky_relu(x)
            #x = 1-F.leaky_relu(1-x)
        else:
            x = torch.clamp(x,0)

def get_DH_params(model):
    
    alpha = model["alpha"]
    theta =  model["theta"]
    radius = model["radius"]
    dists = model["dists"]
    active = model["active"]
    limits = model["limits"]
    
    return alpha, theta, radius, dists, active, limits
